import functools
from typing import Callable, Optional, Any, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

import flax.linen as nn
from flax import struct
from flax.linen.attention import dot_product_attention, Dtype, Array, combine_masks

from haxllm.model.modules import PRNGKey, Shape, default_kernel_init, RMSNorm, make_block_stack
from haxllm.model.parallel import DenseGeneral, Dense, Embed, ShardModule
from haxllm.model.utils import load_config as _load_config
from haxllm.model.efficient_attention import dot_product_attention as dot_product_attention_m
from haxllm.config_utils import RematScanConfigMixin


config_hub = {
    "llama-t": dict(
        hidden_size=1024,
        intermediate_size=2816,
        n_heads=8,
        n_layers=2,
    ),
    "llama-7b": dict(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
    ),
    "llama-13b": dict(
        hidden_size=5120,
        intermediate_size=13824,
        n_heads=40,
        n_layers=40,
    ),
    "llama-30b": dict(
        hidden_size=6656,
        intermediate_size=17920,
        n_heads=52,
        n_layers=60,
    ),
    "llama-65b": dict(
        hidden_size=8192,
        intermediate_size=22016,
        n_heads=64,
        n_layers=80,
    ),
}

def load_config(name, **kwargs):
    if name in config_hub:
        config = config_hub[name]
    else:
        raise ValueError(f"Unknown llama model {name}")
    return _load_config(TransformerConfig, config, **kwargs)


@struct.dataclass
class TransformerConfig(RematScanConfigMixin):
    vocab_size: int = 32000
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 4096
    intermediate_size: int = 11008
    n_heads: int = 32
    n_layers: int = 32
    rms_norm_eps: float = 1e-6
    n_positions: int = 2048
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    decode: bool = False
    memory_efficient: bool = False
    shard: bool = False


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
    seq_len = q.shape[1]
    cos = cos[None, :seq_len, None, :]
    sin = sin[None, :seq_len, None, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
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


class SelfAttention(ShardModule):
    num_heads: int
    max_len: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    decode: bool = False
    memory_efficient: bool = False
    shard: bool = True

    @nn.compact
    def __call__(self, x: Array, mask: Optional[Array] = None):
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
            x = dot_product_attention_m(query, key, value, causal=True, dtype=self.dtype)
        else:
            x = dot_product_attention(query, key, value, mask=mask, dtype=self.dtype)


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

        if not config.memory_efficient:
            attn_mask = nn.make_causal_mask(inputs[..., 0], dtype=jnp.bool_)
        else:
            attn_mask = None

        x = RMSNorm(epsilon=config.rms_norm_eps, dtype=config.dtype, name="ln_1")(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            decode=config.decode,
            memory_efficient=config.memory_efficient,
            shard=config.shard,
            name='attn')(x, attn_mask)
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


def remap_state_dict(state_dict, config: TransformerConfig):
    root = {}
    root['wte'] = {'embedding': state_dict.pop('model.embed_tokens.weight')}
    hidden_size = config.hidden_size
    n_heads = config.n_heads

    for d in range(config.n_layers):
        block_d = {}
        block_d['ln_1'] = {'scale': state_dict.pop(f'model.layers.{d}.input_layernorm.weight')}
        block_d['attn'] = {
            'query': {
                'kernel': state_dict.pop(f'model.layers.{d}.self_attn.q_proj.weight').T.reshape(hidden_size, n_heads, -1),
            },
            'key': {
                'kernel': state_dict.pop(f'model.layers.{d}.self_attn.k_proj.weight').T.reshape(hidden_size, n_heads, -1),
            },
            'value': {
                'kernel': state_dict.pop(f'model.layers.{d}.self_attn.v_proj.weight').T.reshape(hidden_size, n_heads, -1),
            },
            'out': {
                'kernel': state_dict.pop(f'model.layers.{d}.self_attn.o_proj.weight').T.reshape(n_heads, -1, hidden_size),
            },
        }
        block_d['ln_2'] = {'scale': state_dict.pop(f'model.layers.{d}.post_attention_layernorm.weight')}
        block_d['mlp'] = {
            'gate': {'kernel': state_dict.pop(f'model.layers.{d}.mlp.gate_proj.weight').T},
            'up': {'kernel': state_dict.pop(f'model.layers.{d}.mlp.up_proj.weight').T},
            'down': {'kernel': state_dict.pop(f'model.layers.{d}.mlp.down_proj.weight').T},
        }
        root[f'h_{d}'] = block_d

    root['ln_f'] = {'scale': state_dict.pop('model.norm.weight')}
    root['lm_head'] = {'kernel': state_dict.pop('lm_head.weight').T}
    return root