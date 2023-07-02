from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

from flax.core.frozen_dict import unfreeze, freeze
from flax.traverse_util import unflatten_dict, flatten_dict
from flax import linen as nn

from haxllm.utils import load_transformer_params
from haxllm.model.decode import random_sample, beam_search, fix_cache_index


def find_pad_context_length(seq_len, multiple=64):
    # Find the smallest multiple of `multiple` that is larger than seq_len
    return (seq_len + multiple - 1) // multiple * multiple


class TextGenerationPipeline:

    def __init__(self, tokenizer, model, mesh=None, max_len=512, seed=0, two_stage=False, pad_multiple=64):
        r"""
        Initialize the TextGenerationPipeline with given tokenizer, model, and other optional parameters.

        Parameters
        ----------
        tokenizer:
            Tokenizer object that handles text encoding and decoding.
        model:
            Pre-trained model for text generation.
        mesh: jax.sharding.Mesh, default None
            a Mesh object for parallel processing.
            If None, use single device and init the model on CPU.
        max_len: int, default 512
            maximum length of the generated text.
        seed: int, default 0
            random seed for reproducible results.
        two_stage: bool, default False
            flag to enable two-stage decoding.
        pad_multiple: int, default 64
            multiple of padding length for two-stage decoding (to avoid jit recompilation)
        """
        self.seed = seed

        tokenizer.decode(tokenizer("init")['input_ids'])
        self.tokenizer = tokenizer

        self.model = model
        self.mesh = mesh
        self.max_len = max_len
        self.two_stage = two_stage
        self.pad_context = two_stage
        self.pad_multiple = pad_multiple

        self._rng = random.PRNGKey(self.seed)

        self.params = None
        self.cache = None
        self._apply_fn = None

        self._ccache = None

    def init(self, transformer_weight=None):
        self._rng, init_rng = random.split(self._rng)
        input_ids = jnp.zeros((1, self.max_len), dtype=jnp.int32)

        def init_fn(init_rng, input_ids, model):
            state = model.init(init_rng, input_ids=input_ids)
            return state['params'], state['cache']

        mesh = self.mesh
        parallel = mesh is not None
        if parallel:
            device_mesh = mesh_utils.create_device_mesh(mesh, contiguous_submeshes=True)
            mesh = Mesh(devices=device_mesh, axis_names=("X", "Y"))
            load_device = mesh
            print("Init model on {}".format(mesh))

            abs_params, abs_cache = jax.eval_shape(
                partial(init_fn, model=self.model), init_rng, input_ids)
            params_spec = nn.get_partition_spec(abs_params)
            cache_spec = nn.get_partition_spec(abs_cache)

            p_init_fn = pjit(
                partial(init_fn, model=self.model), out_shardings=(params_spec, cache_spec))
            with mesh:
                params, cache = p_init_fn(init_rng, input_ids)
        else:
            print("Init model on CPU")
            load_device = 'cpu'
            p_init_fn = jax.jit(partial(init_fn, model=self.model))

            with jax.default_device(jax.devices("cpu")[0]):
                params, cache = p_init_fn(init_rng, input_ids)

        if transformer_weight:
            params = unfreeze(flatten_dict(params, sep="."))
            params = load_transformer_params(
                params, transformer_weight, lm_head=True, device=load_device)
            params = freeze(unflatten_dict(params, sep="."))

        if not parallel:
            params = jax.device_put(params, jax.devices()[0])
            cache = jax.device_put(cache, jax.devices()[0])

        def apply_fn(params, cache, input_ids, model):
            logits, new_vars = model.apply(
                {"params": params, "cache": cache},
                input_ids=input_ids,
                train=False,
                mutable=["cache"],
            )
            return new_vars['cache'], logits

        p_apply_fn = jax.jit(partial(apply_fn, model=self.model))

        self._apply_fn = p_apply_fn
        self.params = params
        self.cache = cache
    
    def check_init(self):
        if self.params is None or self.cache is None or self._apply_fn is None:
            raise RuntimeError("Please call init() first.")

    def prepare_call_args(self, sentence, max_len):
        self.check_init()
        input_ids = self.tokenizer(sentence)['input_ids']
        max_len = max_len or self.max_len
        if self.two_stage and self.pad_context:
            pad_context = find_pad_context_length(len(input_ids), self.pad_multiple)
        else:
            pad_context = None
        pad_token_id = self.tokenizer.pad_token_id
        return input_ids, {
            "max_len": max_len,
            "two_stage": self.two_stage,
            "pad_token_id": pad_token_id,
            "pad_context": pad_context,
        }

    def greedy_search(self, sentence, max_len=None):
        inputs, kwargs = self.prepare_call_args(sentence, max_len)
        return random_sample(
            inputs, self.tokenizer, self._apply_fn, self.params, self.cache,
            temperature=0.0, **kwargs)
    
    def random_sample(self, sentence, temperature=1.0, top_k=-1, top_p=1.0, max_len=None, rng=None):
        if rng is None:
            self._rng, rng = random.split(self._rng)
        inputs, kwargs = self.prepare_call_args(sentence, max_len)
        return random_sample(
            inputs, self.tokenizer, self._apply_fn, self.params, self.cache,
            temperature=temperature, top_k=top_k, top_p=top_p, rng=rng, **kwargs)
    
    def beam_search(self, sentence, beam_size=1, max_len=None):
        inputs, kwargs = self.prepare_call_args(sentence, max_len)
        return beam_search(
            inputs, self.tokenizer, self._apply_fn, self.params, self.cache,
            n_beams=beam_size, **kwargs)


class ChatPipeline(TextGenerationPipeline):

    def __init__(self, tokenizer, model, mesh=None, max_len=512, seed=0, pad_multiple=64):
        r"""
        Initialize the ChatPipeline with given tokenizer, model, and other optional parameters.

        Parameters
        ----------
        tokenizer:
            Tokenizer object that handles text encoding and decoding.
        model:
            Pre-trained model for text generation.
        mesh: jax.sharding.Mesh, default None
            a Mesh object for parallel processing.
            If None, use single device and init the model on CPU.
        max_len: int, default 512
            maximum length of the generated text.
        seed: int, default 0
            random seed for reproducible results.
        pad_multiple: int, default 64
            multiple of padding length for two-stage decoding (to avoid jit recompilation)
        """
        super().__init__(tokenizer, model, mesh, max_len, seed, True, pad_multiple)
        self.reset_chat_state()
    
    def reset_chat_state(self):
        self._ccache = jax.tree_map(lambda x: jnp.tile(x, [1] * x.ndim), self.cache)
    
    def get_cache_index(self):
        cache = self._ccache['transformer']
        if 'hs' in cache:
            cache_index = int(cache['hs']['attn']['cache_index'][0])
        else:
            cache_index = int(cache['h_0']['attn']['cache_index'])
        return cache_index
    
    def get_next_rng(self):
        self._rng, rng = random.split(self._rng)
        return rng

    def stream_forward(self, input_ids):
        self.check_init()
        cache = self._ccache
        assert input_ids.ndim == 2 and input_ids.shape[0] == 1
        seq_len = input_ids.shape[1]
        if seq_len > 1:
            pad_context = find_pad_context_length(seq_len, self.pad_multiple)
            pad_token_id = self.tokenizer.pad_token_id
            inputs = jnp.full((1, pad_context), pad_token_id, dtype=jnp.int32)
            inputs = inputs.at[:, :seq_len].set(input_ids)
            cache, logits = self._apply_fn(self.params, cache, inputs)
            logits = logits[:, :seq_len]
            cache = fix_cache_index(cache, pad_context - seq_len)
        else:
            cache, logits = self._apply_fn(self.params, cache, input_ids)
        logits = logits.astype(jnp.float32)
        self._ccache = cache
        return logits
