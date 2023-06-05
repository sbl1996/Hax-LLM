import functools
from typing import Tuple

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.parallel import SelfAttention, DenseGeneral, MlpBlock
from haxllm.model.utils import load_config as _load_config
from haxllm.model.gpt2 import (
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
        raise ValueError(f"Unknown lora gpt2 model {name}")

    d = {**kwargs}
    if "attn_lora_r" in d and d["attn_lora_r"] is not None:
        assert len(d["attn_lora_r"]) == 4
        d["attn_lora_r"] = tuple(d["attn_lora_r"])
    if "ffn_lora_r" in d and d["ffn_lora_r"] is not None:
        assert len(d["ffn_lora_r"]) == 2
        d["ffn_lora_r"] = tuple(d["ffn_lora_r"])

    return _load_config(TransformerConfig, config, **d)


@struct.dataclass
class TransformerConfig(BaseTransformerConfig):
    attn_lora_r: Tuple[int] = (0, 0, 0, 0)
    ffn_lora_r: Tuple[int] = (0, 0)
    lora_alpha: int = 1


class TransformerBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        attn_mask = nn.make_causal_mask(inputs[..., 0], dtype=jnp.bool_)

        attn_dense = [
            functools.partial(
                LoraDenseGeneral, r=r, lora_alpha=config.lora_alpha)
            if r > 0 else DenseGeneral
            for r in config.attn_lora_r
        ]

        x = nn.LayerNorm(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_1"
        )(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            dropout_rate=config.attn_pdrop,
            deterministic=self.deterministic,
            decode=config.decode,
            qkv_shard_axes=("X", "Y", None),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            dense_cls=attn_dense,
            name="attn")(x, attn_mask)
        x = nn.Dropout(rate=config.resid_pdrop)(x, deterministic=self.deterministic)
        x = x + inputs

        mlp_dense = [
            functools.partial(
                LoraDenseGeneral, r=r, lora_alpha=config.lora_alpha)
            if r > 0 else DenseGeneral
            for r in config.ffn_lora_r
        ]

        y = nn.LayerNorm(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_2"
        )(x)
        y = MlpBlock(
            activation="gelu_new",
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes1=("X", "Y"),
            shard_axes2=("Y", "X"),
            shard=config.shard,
            dense_cls=mlp_dense,
            name="mlp",
        )(y)
        y = nn.Dropout(rate=config.resid_pdrop)(y, deterministic=self.deterministic)

        if self.scan:
            return x + y, None
        else:
            return x + y


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(config=config, block_cls=TransformerBlock, name="transformer")(
            inputs=inputs, train=train
        )

        batch_size = inputs.shape[0]
        seq_len = jnp.not_equal(inputs, config.pad_token_id).sum(-1) - 1
        x = x[jnp.arange(batch_size), seq_len]

        x = nn.Dropout(rate=config.cls_pdrop)(x, not train)
        x = DenseGeneral(
            config.num_labels,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name="score",
        )(x)
        return x


class TransformerLMHeadModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, train=False):
        config = self.config
        x = TransformerModel(config=config, block_cls=TransformerBlock, name="transformer")(
            inputs=inputs, train=train
        )

        x = DenseGeneral(
            config.vocab_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            kernel_init=config.kernel_init,
            name="lm_head",
        )(x)
        return x