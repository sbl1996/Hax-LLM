import functools
from typing import Callable, Optional, Any, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

import flax.linen as nn
from flax import struct
from flax.linen import initializers
from flax.linen.attention import Dtype, Array, combine_masks

from haxllm.model.modules import PRNGKey, Shape, default_kernel_init, RMSNorm, make_block_stack
from haxllm.model.parallel import DenseGeneral, Dense, Embed, ShardModule, dot_product_attention, PrefixEmbed
from haxllm.model.utils import load_config as _load_config
from haxllm.model.efficient_attention import dot_product_attention as dot_product_attention_m
from haxllm.config_utils import RematScanConfigMixin
from haxllm.model.llama import config_hub, remap_state_dict, TransformerConfig as BaseTransformerConfig


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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32)[: (dim // 2)] / dim))
    t = np.arange(end, dtype=np.float32)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    freqs = np.concatenate((freqs, freqs), axis=-1)
    cos, sin = np.cos(freqs), np.sin(freqs)
    return jnp.array(cos, dtype=dtype), jnp.array(sin, dtype=dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # x: (batch_size, seq_len, num_heads, head_dim)
    # cos, sin: (seq_len, head_dim)
    q_len = q.shape[1]
    kv_len = k.shape[1]
    prefix_len = kv_len - q_len

    cos_k = cos[None, :kv_len, None, :]
    sin_k = sin[None, :kv_len, None, :]
    k = (k * cos_k) + (rotate_half(k) * sin_k)

    cos_q = cos_k[:, prefix_len:]
    sin_q = sin_k[:, prefix_len:]
    q = (q * cos_q) + (rotate_half(q) * sin_q)
    return q, k


class MlpBlock(ShardModule):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        dense = functools.partial(
            # p_remat(Dense),  # Hack: remat here with less memory and same speed
            Dense,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            shard=config.shard,
        )

        actual_out_dim = inputs.shape[-1]
        g = dense(
            features=config.intermediate_size,
            shard_axes={"kernel": ("X", "Y")},
            name="gate")(inputs)
        g = nn.silu(g)
        # g = self.with_sharding_constraint(
        #     g, ("X", None, "Y"))
        x = g * dense(
            features=config.intermediate_size,
            shard_axes={"kernel": ("X", "Y")},
            name="up")(inputs)
        # x = self.with_sharding_constraint(
        #     x, ("X", None, "Y"))
        x = dense(
            features=actual_out_dim,
            shard_axes={"kernel": ("Y", "X")},
            name="down")(x)
        # x = self.with_sharding_constraint(
        #     x, ("X", None, "Y"))
        return x


class RoPESelfAttention(ShardModule):
    num_heads: int
    max_len: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    decode: bool = False
    zero_init: bool = False
    memory_efficient: bool = False
    shard: bool = True

    @nn.compact
    def __call__(self, x: Array, mask: Optional[Array] = None,
                 prefix_key_value: Optional[Array] = None):
        features = x.shape[-1]
        assert features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = features // self.num_heads
        dense = functools.partial(
            DenseGeneral,
            # p_remat(DenseGeneral),
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
            shard_axes={"kernel": ("X", "Y", None)},
            shard=self.shard,
        )

        # qkv_constraint = lambda x: x
        qkv_constraint = functools.partial(
            self.with_sharding_constraint, axes=("X", None, "Y", None))

        query, key, value = (
            qkv_constraint(dense(name='query')(x)),
            qkv_constraint(dense(name='key')(x)),
            qkv_constraint(dense(name='value')(x)),
        )

        if prefix_key_value is not None:
            key = jnp.concatenate([prefix_key_value[0], key], axis=1)
            value = jnp.concatenate([prefix_key_value[1], value], axis=1)
            prefix_len = prefix_key_value[0].shape[1]
        else:
            prefix_len = None

        cos, sin = precompute_freqs_cis(dim=head_dim, end=self.max_len, dtype=self.dtype)

        if not self.decode:
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        else:
            is_initialized = self.has_variable('cache', 'cached_key')
            zero_init = jnp.zeros
            if self.shard:
                zero_init = nn.with_partitioning(zero_init, (None, None, "Y", None))
            cached_key = self.variable('cache', 'cached_key',
                                       zero_init, key.shape, key.dtype)
            cached_value = self.variable('cache', 'cached_value',
                                         zero_init, value.shape, value.dtype)
            cache_index = self.variable('cache', 'cache_index',
                                        lambda: jnp.array(0, dtype=jnp.int32))
            if not is_initialized:
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
            else:
                *batch_dims, max_length, num_heads, depth_per_head = (
                    cached_key.value.shape)
                # shape check of cached keys against query input
                expected_shape = tuple(batch_dims) + \
                    (1, num_heads, depth_per_head)
                if expected_shape != query.shape:
                    raise ValueError('Autoregressive cache shape error, '
                                     'expected query shape %s instead got %s.' %
                                     (expected_shape, query.shape))
                # update key, value caches with our new 1d spatial slices
                cur_index = cache_index.value
                pos_index = jnp.array([cur_index], dtype=jnp.int32)
                cos, sin = cos[pos_index], sin[pos_index]
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(
                    cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                                     tuple(batch_dims) + (1, 1, max_length)))

        if self.memory_efficient:
            assert mask is None, 'Masking is not supported for memory efficient attention, default to causal attention.'
            assert not self.zero_init, 'Zero init is not supported for memory efficient attention'
            x = dot_product_attention_m(query, key, value, causal=True, dtype=self.dtype)
        else:
            if self.zero_init:
                prefix_gate = self.param(
                    "prefix_gate", initializers.zeros, (self.num_heads,), jnp.float32)
                prefix_gate = prefix_gate[None, :, None, None]
            else:
                prefix_gate = None
            x = dot_product_attention(query, key, value, prefix_gate,
                                      mask=mask, prefix_len=prefix_len, dtype=self.dtype)

        # x = self.with_sharding_constraint(
        #     x, axes=("X", None, "Y", None))
        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            shard_axes={"kernel": ("Y", None, "X")},
            shard=self.shard,
            name='out',
        )(x)
        # out = self.with_sharding_constraint(
        #     out, ("X", None, "Y"))
        return out


class TransformerBlock(nn.Module):
    config: TransformerConfig
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

        if not config.memory_efficient:
            prefix_len = config.pre_seq_len
            kv_len = config.pre_seq_len + inputs.shape[1]
            idxs = jnp.arange(kv_len, dtype=jnp.int32)
            mask = (idxs[:, None] > idxs[None, :])[prefix_len:, :]
            mask = jnp.broadcast_to(mask, (inputs.shape[0], 1, kv_len - prefix_len, kv_len))
        else:
            mask = None

        x = RMSNorm(epsilon=config.rms_norm_eps, dtype=config.dtype, name="ln_1")(inputs)
        x = RoPESelfAttention(
            num_heads=config.n_heads,
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            decode=config.decode,
            memory_efficient=config.memory_efficient,
            shard=config.shard,
            zero_init=config.zero_init_prefix_attn,
            name='attn')(x, mask, prefix_key_value)
        x = x + inputs

        y = RMSNorm(epsilon=config.rms_norm_eps, dtype=config.dtype, name="ln_2")(x)
        y = MlpBlock(config=config, name='mlp')(y)
        if self.scan:
            return x + y, None
        else:    
            return x + y


class TransformerModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, train):
        config = self.config
        remat = config.remat or config.remat_scan

        embed_layer = Embed if remat else Embed
        x = embed_layer(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": (None, "Y")},
            shard=config.shard,
            name='wte'
        )(inputs)

        x = make_block_stack(TransformerBlock, config.n_layers, config)(x, train)

        norm_layer = RMSNorm if remat else RMSNorm
        x = norm_layer(epsilon=config.rms_norm_eps, dtype=config.dtype, name='ln_f')(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(config=config, name='transformer')(inputs, train)

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