from typing import Union, List, Tuple

import warnings
from functools import partial

from tokenizers import Tokenizer

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.experimental import mesh_utils

from flax.core.frozen_dict import unfreeze, freeze
from flax.traverse_util import unflatten_dict, flatten_dict
from flax import linen as nn

from haxllm.utils import load_transformer_params
from haxllm.model.decode import random_sample, beam_search, fix_cache_index, batch_random_sample, add_batch_dim
from haxllm.gconfig import get_seed


def find_pad_context_length(seq_len, multiple=64):
    # Find the smallest multiple of `multiple` that is larger than seq_len
    return (seq_len + multiple - 1) // multiple * multiple


def init_mesh(mesh):
    if mesh is None or isinstance(mesh, Mesh):
        return mesh
    device_mesh = mesh_utils.create_device_mesh(mesh)
    mesh = Mesh(devices=device_mesh, axis_names=("X", "Y"))
    return mesh


def init_fn(init_rng, input_ids, model, cache_only):
    state = model.init(init_rng, input_ids=input_ids)
    state = freeze(state)
    if cache_only:
        return state['cache']
    return state['params'], state['cache']


class TextGenerationPipeline:

    def __init__(self, tokenizer: Tokenizer, model, max_len=512, seed=None, rng=None,
                 two_stage=True, pad_multiple=512, temperature=1.0, top_k=-1,
                 top_p=1.0, repetition_penalty=1.0, min_p=0.0,
                 max_new_tokens=None, stop_token_ids=None, verbose=True):
        r"""
        Initialize the TextGenerationPipeline with given tokenizer, model, and other optional parameters.

        Parameters
        ----------
        tokenizer:
            Tokenizer object that handles text encoding and decoding.
        model:
            Pre-trained model for text generation.
        max_len: int, default 512
            maximum length of the generated text.
        seed: int, default None
            random seed for reproducible results.
            If None, use seed from gconfig.
        two_stage: bool, default False
            flag to enable two-stage decoding.
        pad_multiple: int, default 512
            multiple of padding length for two-stage decoding (to avoid jit recompilation)
        rng: jax.random.PRNGKey, default None
            random number generator for reproducible results.
        temperature: float, default 1.0
            temperature for random sampling.
        top_k: int, default -1
            number of top-k tokens to sample from.
        top_p: float, default 1.0
            cumulative probability for top-p sampling.
        repetition_penalty: float, default 1.0
            penalty for repeated tokens.
        max_new_tokens: int, default None
            maximum number of new tokens to generate.
        stop_token_ids: list of int, default None
            list of token ids to stop decoding.
        verbose: bool, default True
            flag to enable verbose output.
        """
        if seed is None:
            seed = get_seed()
        self.seed = seed

        tokenizer.decode(tokenizer("init")['input_ids'])
        self.tokenizer = tokenizer

        self.model = model
        self.max_len = max_len
        self.two_stage = two_stage
        self.pad_context = two_stage
        self.pad_multiple = pad_multiple
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.stop_token_ids = stop_token_ids
        self.verbose = verbose

        if rng is None:
            self._rng = random.PRNGKey(self.seed)
        else:
            self._rng = rng

        self.params = None
        self.cache = None
        self._apply_fn = None

        # inspection
        self._apply_fn_i = None

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def init_without_params(self, mesh=None):
        return self.init(mesh=mesh, no_params=True)

    def init(self, transformer_weight=None, mesh=None, no_params=False):
        self._rng, init_rng = random.split(self._rng)
        input_ids = jnp.zeros((1, self.max_len), dtype=jnp.int32)

        mesh = init_mesh(mesh)
        parallel = mesh is not None
        if parallel:
            load_device = mesh
            self.print("Init model on {}".format(mesh))

            abs_params, abs_cache = jax.eval_shape(
                partial(init_fn, model=self.model, cache_only=False), init_rng, input_ids)
            cache_sharding = nn.get_sharding(abs_cache, mesh)

            p_init_fn = jax.jit(
                partial(init_fn, model=self.model, cache_only=True), out_shardings=cache_sharding)
            with mesh:
                cache = p_init_fn(init_rng, input_ids)
            params = abs_params
        else:
            self.print("Init model on CPU")
            load_device = 'cpu'
            p_init_fn = jax.jit(partial(init_fn, model=self.model, cache_only=False))

            with jax.default_device(jax.devices("cpu")[0]):
                params, cache = p_init_fn(init_rng, input_ids)

        if transformer_weight:
            params = unfreeze(flatten_dict(params, sep="."))
            params = load_transformer_params(
                params, transformer_weight, lm_head=True, device=load_device, verbose=self.verbose)
            params = freeze(unflatten_dict(params, sep="."))

        if not parallel:
            if not no_params:
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
        self.cache = cache
        if not no_params:
            self.params = params
    
    def init_inspection(self):
        if self._apply_fn_i is not None:
            return
        def apply_fn(params, cache, input_ids, model):
            logits, new_vars = model.apply(
                {"params": params, "cache": cache},
                input_ids=input_ids,
                train=False,
                mutable=["cache", "intermediates"],
            )
            return new_vars['cache'], logits, new_vars['intermediates']

        p_apply_fn = jax.jit(partial(apply_fn, model=self.model))

        self._apply_fn_i = p_apply_fn

    def set_params(self, params):
        self.params = params
    
    def unset_params(self):
        self.params = None

    def check_init(self):
        if self.params is None or self.cache is None or self._apply_fn is None:
            raise RuntimeError("Please call init() first.")

    def prepare_call_args(self, inputs):
        self.check_init()
        if isinstance(inputs, str):
            input_ids = self.tokenizer(inputs)['input_ids']
            decode = True
        else:
            input_ids = inputs
            decode = False
        if self.two_stage and self.pad_context:
            pad_context = find_pad_context_length(len(input_ids), self.pad_multiple)
        else:
            pad_context = None
        pad_token_id = self.tokenizer.pad_token_id
        return input_ids, {
            "two_stage": self.two_stage,
            "pad_token_id": pad_token_id,
            "pad_context": pad_context,
            "decode": decode,
        }

    def greedy_search(self, inputs, max_len=None, max_source_length=None, stop_token_ids=None, padding_left=None):
        return self.random_sample(inputs, temperature=0.0, max_len=max_len, max_source_length=max_source_length,
                                  stop_token_ids=stop_token_ids, padding_left=padding_left)

    def random_sample(
            self, inputs, temperature=None, top_k=None, top_p=None, min_p=None,
            repetition_penalty=None, max_len=None, rng=None, max_source_length=None,
            stop_token_ids=None, padding_left=None, max_new_tokens=None):
        temperature = self.temperature if temperature is None else temperature
        stop_token_ids = self.stop_token_ids if stop_token_ids is None else stop_token_ids

        top_k = self.top_k if top_k is None else top_k
        top_p = self.top_p if top_p is None else top_p
        min_p = self.min_p if min_p is None else min_p
        repetition_penalty = self.repetition_penalty if repetition_penalty is None else repetition_penalty
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens

        max_len = max_len or self.max_len

        is_greedy = temperature == 0.0 or top_k == 1

        if padding_left is None:
            padding_left = self.tokenizer.padding_side == "left"
        if padding_left:
            if not getattr(self.model.config, "padding_left", False):
                warnings.warn(
                    "provided padding_left={}, tokenizer.padding_side={}, but model does not have "
                    "config.padding_left=True. This may cause errors.".format(
                        padding_left, self.tokenizer.padding_side
                    ),
                )

        if isinstance(inputs, str) or (isinstance(inputs, np.ndarray) and inputs.ndim == 1):
            if not is_greedy and rng is None:
                self._rng, rng = random.split(self._rng)
            inputs, kwargs = self.prepare_call_args(inputs)
            return random_sample(
                inputs, self.tokenizer, self._apply_fn, self.params, self.cache,
                temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                repetition_penalty=repetition_penalty, rng=rng, max_len=max_len,
                max_new_tokens=max_new_tokens, stop_token_ids=stop_token_ids, **kwargs)
        elif isinstance(inputs, list) or (isinstance(inputs, np.ndarray) and inputs.ndim == 2):
            if not is_greedy and rng is None:
                self._rng, rng = random.split(self._rng)
            pad_token_id = self.tokenizer.pad_token_id
            if isinstance(inputs, list):
                decode = True
                if max_source_length is None:
                    input_ids = self.tokenizer(inputs)['input_ids']
                else:
                    input_ids = self.tokenizer(inputs, max_length=max_source_length, truncation=True)['input_ids']
                pad_len = max(len(s) for s in input_ids)
                if padding_left:
                    input_ids = [[pad_token_id] * (pad_len - len(s)) + s for s in input_ids]
                else:
                    input_ids = [s + [pad_token_id] * (pad_len - len(s)) for s in input_ids]
                input_ids = np.array(input_ids, dtype=np.int32)
            else:
                input_ids = inputs
                decode = False
            max_len = max_len or self.max_len
            outputs = batch_random_sample(
                input_ids, self._apply_fn, self.params, self.cache,
                max_len=max_len, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                rng=rng, pad_token_id=pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens, stop_token_ids=stop_token_ids, padding_left=padding_left)
            if decode:
                outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return outputs
        else:
            raise TypeError("inputs must be a string or a list of strings or np.ndarray.")
    
    def beam_search(self, sentence, beam_size=1, max_len=None):
        inputs, kwargs = self.prepare_call_args(sentence, max_len)
        return beam_search(
            inputs, self.tokenizer, self._apply_fn, self.params, self.cache,
            n_beams=beam_size, **kwargs)

    def forward(self, input_ids, inspect=False):
        if inspect:
            assert self._apply_fn_i is not None, "call `init_inspection` before calling `forward` with inspect=True"
        apply_fn = self._apply_fn if not inspect else self._apply_fn_i
        batch_size, max_source_length = input_ids.shape
        cache = jax.tree.map(lambda x: add_batch_dim(x, batch_size), self.cache)
        input_ids = jnp.asarray(input_ids)
        return apply_fn(self.params, cache, input_ids)


class ChatPipeline(TextGenerationPipeline):

    def __init__(self, tokenizer, model, max_len=512, seed=0, pad_multiple=512, **kwargs):
        r"""
        Initialize the ChatPipeline with given tokenizer, model, and other optional parameters.

        Parameters
        ----------
        tokenizer:
            Tokenizer object that handles text encoding and decoding.
        model:
            Pre-trained model for text generation.
        max_len: int, default 512
            maximum length of the generated text.
        seed: int, default 0
            random seed for reproducible results.
        pad_multiple: int, default 512
            multiple of padding length for two-stage decoding (to avoid jit recompilation)
        """
        super().__init__(tokenizer, model, max_len, seed, two_stage=True, pad_multiple=pad_multiple, **kwargs)
        self.reset_chat_state()
        self.conv_template = None
    
    def reset_chat_state(self):
        self._ccache = jax.tree.map(lambda x: jnp.tile(x, [1] * x.ndim), self.cache)
    
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


# TODO: refactor
def apply_with_pad(apply_fn, params, cache, input_ids, pad_multiple, pad_token_id):
    seq_len = input_ids.shape[1]
    pad_context = find_pad_context_length(seq_len, pad_multiple)
    inputs = jnp.full((1, pad_context), pad_token_id, dtype=jnp.int32)
    inputs = inputs.at[:, :seq_len].set(input_ids)
    cache, logits = apply_fn(params, cache, inputs)
    logits = logits[:, :seq_len]
    cache = fix_cache_index(cache, pad_context - seq_len)
    return cache, logits
