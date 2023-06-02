import functools

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.parallel import MlpBlock
from haxllm.model.modules import make_block_stack
from haxllm.model.utils import load_config as _load_config, truncated_normal_init
from haxllm.model.bert import (
    config_hub,
    remap_state_dict,
    TransformerConfig as BaseTransformerConfig,
)
from haxllm.model.ptuning.modules import PrefixEmbed, SelfAttention


def load_config(name, **kwargs):
    if name in config_hub:
        config = config_hub[name]
    else:
        raise ValueError(f"Unknown ptuning bert model {name}")
    return _load_config(TransformerConfig, config, **kwargs)


@struct.dataclass
class TransformerConfig(BaseTransformerConfig):
    pre_seq_len: int = 0
    prefix_projection: bool = False
    prefix_hidden_size: int = 512
    zero_init_prefix_attn: bool = False


class TransformerBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool
    scan: bool = False

    @nn.compact
    def __call__(self, x):
        inputs, attn_mask = x
        config = self.config

        prefix_key_value = PrefixEmbed(
            seq_len=config.pre_seq_len,
            features=(config.n_heads, config.hidden_size // config.n_heads),
            projection=config.prefix_projection,
            prefix_features=config.prefix_hidden_size,
            dtype=config.dtype,
            name="prefix",
        )(inputs)

        if attn_mask is not None:
            # (B, H, Q, KV)
            attn_mask_ = jnp.concatenate(
                (
                    jnp.ones(
                        (attn_mask.shape[0], config.pre_seq_len), dtype=attn_mask.dtype
                    ),
                    attn_mask,
                ),
                axis=-1,
            )
            attn_mask_ = attn_mask_[:, None, None, :]
        else:
            attn_mask_ = None

        x = SelfAttention(
            num_heads=config.n_heads,
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            dropout_rate=config.attn_pdrop,
            deterministic=self.deterministic,
            broadcast_dropout=False,
            qkv_shard_axes=("X", "Y", None),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            zero_init=config.zero_init_prefix_attn,
            name="attn",
        )(inputs, attn_mask_, prefix_key_value)
        x = nn.Dropout(rate=config.resid_pdrop)(x, deterministic=self.deterministic)
        x = x + inputs
        x = nn.LayerNorm(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_1"
        )(x)

        y = MlpBlock(
            intermediate_size=config.intermediate_size,
            activation="gelu",
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            shard_axes1=("X", "Y"),
            shard_axes2=("Y", "X"),
            shard=config.shard,
            name="mlp",
        )(x)
        y = nn.Dropout(rate=config.resid_pdrop)(y, deterministic=self.deterministic)
        y = x + y
        y = nn.LayerNorm(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_2"
        )(y)

        if self.scan:
            return (y, attn_mask), None
        else:
            return y, attn_mask


class TransformerModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, position_ids=None, train=False):
        config = self.config

        if position_ids is None:
            position_ids = jnp.arange(0, inputs.shape[-1], dtype=jnp.int32)[None]
        position_ids = position_ids + config.pre_seq_len
        type_token_ids = jnp.zeros_like(position_ids, dtype=jnp.int32)

        embed_layer = functools.partial(
            nn.Embed, dtype=config.dtype, param_dtype=config.param_dtype
        )

        inputs_embeds = embed_layer(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            name="word_embeddings",
        )(inputs)
        position_embeds = embed_layer(
            num_embeddings=config.n_positions,
            features=config.hidden_size,
            name="position_embeddings",
        )(position_ids)
        token_type_embeds = embed_layer(
            num_embeddings=config.type_vocab_size,
            features=config.hidden_size,
            name="token_type_embeddings",
        )(type_token_ids)

        x = inputs_embeds + position_embeds + token_type_embeds

        x = nn.LayerNorm(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln"
        )(x)

        x = nn.Dropout(rate=config.embd_pdrop)(x, not train)

        block_fn = functools.partial(TransformerBlock, deterministic=not train)
        x = make_block_stack(block_fn, config.n_layers, config)((x, attn_mask), train)[
            0
        ]

        if config.pooler:
            x = nn.Dense(
                config.hidden_size,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
                kernel_init=truncated_normal_init(stddev=config.initializer_range),
                bias_init=config.bias_init,
                name="pooler_fc",
            )(x[:, 0])
            x = jnp.tanh(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(config=config, name="transformer")(
            inputs=inputs, attn_mask=attn_mask, train=train
        )

        if not config.pooler:
            x = x[:, 0]

        x = nn.Dropout(rate=config.cls_pdrop)(x, not train)
        x = nn.Dense(
            config.num_labels,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=nn.initializers.normal(stddev=config.initializer_range),
            bias_init=config.bias_init,
            name="score",
        )(x)
        return x
