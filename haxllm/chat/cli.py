import os
import time
from functools import partial
import importlib

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

import jax_smi

from flax.core.frozen_dict import unfreeze, freeze
from flax.traverse_util import unflatten_dict, flatten_dict
from flax import linen as nn

from haxllm.utils import load_transformer_params
from haxllm.model.decode import random_sample, beam_search, chat

import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer

from haxllm.chat.conversation import get_conv_template



def find_pad_context_length(seq_len, multiple=64):
    # Find the smallest multiple of `multiple` that is larger than seq_len
    return (seq_len + multiple - 1) // multiple * multiple


class TextGenerationPipeline:
    
    def __init__(self, tokenizer, model, mesh=None, max_len=512, seed=0, two_stage=False, pad_context=None, pad_multiple=64):
        self.seed = seed

        tokenizer.decode(tokenizer("init")['input_ids'])
        self.tokenizer = tokenizer

        self.model = model
        self.mesh = mesh
        self.max_len = max_len
        self.two_stage = two_stage
        if two_stage:
            pad_context = True
        self.pad_context = pad_context
        self.pad_multiple = pad_multiple

        self._rng = random.PRNGKey(self.seed)

        self.params = None
        self.cache = None
        self._apply_fn = None

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
    
    def random_sample(self, sentence, temperature=1.0, topk=1, max_len=None, rng=None):
        if rng is None:
            self._rng, rng = random.split(self._rng)
        inputs, kwargs = self.prepare_call_args(sentence, max_len)
        return random_sample(
            inputs, self.tokenizer, self._apply_fn, self.params, self.cache,
            temperature=temperature, topk=topk, rng=rng, **kwargs)
    
    def beam_search(self, sentence, beam_size=1, max_len=None):
        inputs, kwargs = self.prepare_call_args(sentence, max_len)
        return beam_search(
            inputs, self.tokenizer, self._apply_fn, self.params, self.cache,
            n_beams=beam_size, **kwargs)


    def chat(self, max_len=None, temperature=1.0, topk=1, rng=None):
        max_len = max_len or self.max_len
        if rng is None:
            self._rng, rng = random.split(self._rng)
        conv = get_conv_template("vicuna_v1.1")
        chat(conv, self.tokenizer, self._apply_fn, self.params, self.cache, max_len, temperature, topk, rng=rng)


@hydra.main(version_base=None, config_path="../../configs/chat", config_name="base")
def chat_app(cfg: DictConfig) -> None:
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.initialize_cache(os.path.expanduser("~/jax_cache"))

    start = time.time()
    jax_smi.initialise_tracking()

    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    random_seed = cfg.seed
    tokenizer_name = model_config.pop("tokenizer")

    checkpoint = getattr(cfg, "checkpoint", None)
    if checkpoint is None:
        raise RuntimeError("Please specify a checkpoint to load using checkpoint==/path/to/ckpt_file")
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    print(f"Loading tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.decode(tokenizer("init")['input_ids'])
    print("Load tokenizer {}".format(time.time() - start))

    mesh = getattr(cfg, "mesh", None)
    parallel = mesh is not None
    mod = importlib.import_module("haxllm.model." + model_config.pop("family"))

    config = getattr(mod, "load_config")(
        dtype=jnp.dtype(cfg.dtype),
        param_dtype=jnp.dtype(cfg.param_dtype),
        **model_config,
        decode=True,
        shard=parallel,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.pad_token_id
    # tokenizer.padding_side = "right"

    model = getattr(mod, "TransformerLMHeadModel")(config)

    print("Load config {}".format(time.time() - start))

    pipeline = TextGenerationPipeline(
        tokenizer, model, mesh=mesh, max_len=cfg.max_len, seed=random_seed)

    pipeline.init(transformer_weight=checkpoint)

    pipeline.chat(max_len=cfg.max_len, temperature=cfg.temperature, topk=cfg.topk)


if __name__ == "__main__":
    chat_app()