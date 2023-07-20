import functools
from typing import Tuple

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.modules import RMSNorm
from haxllm.model.parallel import GLUMlpBlock, DenseGeneral, SelfAttention
from haxllm.model.utils import load_config as _load_config
from haxllm.model.chatglm2 import (
    config_hub,
    remap_state_dict,
    TransformerConfig as BaseTransformerConfig,
    TransformerModel,
)
from haxllm.model.lora.modules import DenseGeneral as LoraDenseGeneral


def load_config(name, **kwargs):
    if name in config_hub:
        config = config_hub[name]
    else:
        raise ValueError(f"Unknown lora chatglm2 model {name}")

    d = {**kwargs}
    if "attn_lora_r" in d and d["attn_lora_r"] is not None:
        assert len(d["attn_lora_r"]) == 4
        d["attn_lora_r"] = tuple(d["attn_lora_r"])
    if "ffn_lora_r" in d and d["ffn_lora_r"] is not None:
        assert len(d["ffn_lora_r"]) == 3
        d["ffn_lora_r"] = tuple(d["ffn_lora_r"])

    return _load_config(TransformerConfig, config, **d)


@struct.dataclass
class TransformerConfig(BaseTransformerConfig):
    attn_lora_r: Tuple[int, int, int, int] = (8, 8, 0, 0)
    ffn_lora_r: Tuple[int, int, int] = (0, 0, 0)
    lora_alpha: int = 1



# TODO: skip apply_query_key_layer_scaling, same as query_key_layer_scaling_coeff in chatglm
# no difference in inference (forward), but may be in training (backward)
class TransformerBlock(nn.Module):
    config: TransformerConfig
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):

        config = self.config

        if config.memory_efficient_attention or config.decode:
            mask = None
        else:
            mask = nn.make_causal_mask(inputs[..., 0], dtype=jnp.bool_)

        attn_dense = [
            functools.partial(
                LoraDenseGeneral, r=r, lora_alpha=config.lora_alpha)
            if r > 0 else DenseGeneral
            for r in config.attn_lora_r
        ]

        x = RMSNorm(epsilon=config.layer_norm_epsilon,
                    dtype=config.dtype, name="ln_1")(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            multi_query_groups=config.num_groups,
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            qkv_bias=True,
            out_bias=False,
            decode=config.decode,
            rope=True,
            query_shard_axes=("X", "Y", None),
            kv_shard_axes=("X", None, "Y"),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            shard_cache=config.shard_cache,
            dense_cls=attn_dense,
            name="attn")(x, mask)

        x = x + inputs

        mlp_dense = [
            functools.partial(
                LoraDenseGeneral, r=r, lora_alpha=config.lora_alpha)
            if r > 0 else DenseGeneral
            for r in config.ffn_lora_r
        ]

        y = RMSNorm(epsilon=config.layer_norm_epsilon,
                    dtype=config.dtype, name="ln_2")(x)
        y = GLUMlpBlock(
            intermediate_size=config.intermediate_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            use_bias=False,
            shard_axes1=("X", "Y"),
            shard_axes2=("Y", "X"),
            shard=config.shard,
            dense_cls=mlp_dense,
            name="mlp")(y)

        y = x + y

        if self.scan:
            return y, None
        else:
            return y


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(
            config=config, block_cls=TransformerBlock, name="transformer")(inputs, train)

        batch_size = inputs.shape[0]
        seq_len = jnp.not_equal(inputs, config.pad_token_id).sum(-1) - 1
        x = x[jnp.arange(batch_size), seq_len]

        x = DenseGeneral(
            config.num_labels,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name="score")(x)
        return x


class TransformerLMHeadModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, input_ids, train=False):
        config = self.config
        x = TransformerModel(
            config=config, block_cls=TransformerBlock, name="transformer")(
            inputs=input_ids, train=train)

        if config.decode:
            shard_axes = {"kernel": ("Y", None)}
        else:
            # shard output in training to avoid out of memory
            shard_axes = {'kernel': (None, 'Y')}

        x = DenseGeneral(
            config.vocab_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            shard_axes=shard_axes,
            shard=config.shard,
            name="lm_head")(x)
        return x
