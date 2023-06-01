import functools
from typing import Optional, Any, Callable, Tuple

import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from flax import struct

from haxllm.model.parallel import SelfAttention, Dense, Embed, MlpBlock, PrefixEmbed
from haxllm.model.utils import load_config as _load_config
from haxllm.model.modules import make_block_stack
from haxllm.model.gpt2 import config_hub, remap_state_dict, TransformerConfig as BaseTransformerConfig


def load_config(name, **kwargs):
    if name in config_hub:
        config = config_hub[name]
    else:
        raise ValueError(f"Unknown llama model {name}")
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
    def __call__(self, inputs):
        config = self.config

        prefix_key_value = PrefixEmbed(
            seq_len=config.pre_seq_len,
            projection=config.prefix_projection,
            prefix_features=config.prefix_hidden_size,
            features=(config.n_heads, config.hidden_size // config.n_heads),
            dtype=config.dtype,
            name="prefix"
        )(inputs)

        prefix_len = config.pre_seq_len
        kv_len = config.pre_seq_len + inputs.shape[1]
        idxs = jnp.arange(kv_len, dtype=jnp.int32)
        mask = (idxs[:, None] > idxs[None, :])[prefix_len:, :]
        mask = jnp.broadcast_to(mask, (inputs.shape[0], 1, kv_len - prefix_len, kv_len))

        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name='ln_1')(inputs)
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
            zero_init=config.zero_init_prefix_attn,
            shard=config.shard,
            name='attn')(x, mask, prefix_key_value)
        x = nn.Dropout(rate=config.resid_pdrop)(x, deterministic=self.deterministic)
        x = x + inputs

        y = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name='ln_2')(x)
        y = MlpBlock(
            activation='gelu_new',
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes1=("X", "Y"),
            shard_axes2=("Y", "X"),
            shard=config.shard,
            name='mlp')(y)
        y = nn.Dropout(rate=config.resid_pdrop)(y, deterministic=self.deterministic)

        if self.scan:
            return x + y, None
        else:    
            return x + y


class TransformerModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, train):
        config = self.config
        remat = config.remat or config.remat_scan

        embed_layer = nn.remat(Embed) if remat else Embed
        embed_layer = functools.partial(
            embed_layer,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": (None, "Y")},
            shard=config.shard,
        )

        position_ids = jnp.arange(0, inputs.shape[-1], dtype=jnp.int32)[None]
        position_ids = position_ids + config.pre_seq_len
        if config.decode:
            is_initialized = self.has_variable('cache', 'cache_index')
            cache_index = self.variable('cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.uint32))
            if is_initialized:
                i = cache_index.value
                cache_index.value = i + 1
                position_ids = jnp.tile(i, (inputs.shape[0], 1))

        inputs_embeds = embed_layer(
            num_embeddings=config.vocab_size, name='wte')(inputs)
        position_embeds = embed_layer(
            num_embeddings=config.n_positions, name='wpe')(position_ids)

        x = inputs_embeds + position_embeds

        dropout_layer = nn.remat(nn.Dropout, static_argnums=(2,)) if remat else nn.Dropout
        x = dropout_layer(rate=config.embd_pdrop)(x, not train)

        block_fn = functools.partial(
            TransformerBlock, deterministic=not train)
        x = make_block_stack(block_fn, config.n_layers, config)(x, train)

        norm_layer = nn.remat(nn.LayerNorm) if remat else nn.LayerNorm
        x = norm_layer(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_f')(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(config=config, name='transformer')(inputs=inputs, train=train)

        batch_size = inputs.shape[0]
        seq_len = (jnp.not_equal(inputs, config.pad_token_id).sum(-1) - 1)
        x = x[jnp.arange(batch_size), seq_len]

        x = Dense(
            config.num_labels,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name='score')(x)
        return x


class TransformerLMHeadModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, train=False):
        config = self.config
        x = TransformerModel(config=config, name='transformer')(inputs=inputs, train=train)

        x = Dense(
            config.vocab_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            name='lm_head')(x)
        return x