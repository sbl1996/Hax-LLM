import functools
from typing import Any, Callable, Optional

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

from haxllm.model.modules import Dtype, Array, PRNGKey, Shape, default_kernel_init, DenseGeneral, dot_product_attention, RMSNorm
from haxllm.model.utils import load_config as _load_config

config_hub = {
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
class TransformerConfig:
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
    remat: bool = False
    scan: bool = False
    remat_scan: bool = False

    def scan_layers(self):
        return self.n_layers if self.scan else 0

    def remat_scan_lengths(self):
        return (self.n_layers, 1) if self.remat_scan else None

    def scan_lengths(self):
        if self.remat_scan:
            return self.remat_scan_lengths()
        elif self.scan:
            return self.scan_layers()
        else:
            return None


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



class MlpBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        dense = functools.partial(
            nn.Dense,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
        )

        actual_out_dim = inputs.shape[-1]
        g = nn.silu(dense(config.intermediate_size, name="gate")(inputs))
        x = g * dense(config.intermediate_size, name="up")(inputs)
        x = dense(actual_out_dim, name="down")(x)
        return x


class SelfAttention(nn.Module):
    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    decode: bool = False

    @nn.compact
    def __call__(self, x: Array, mask: Optional[Array] = None):
        seq_len = x.shape[1]
        features = x.shape[-1]
        assert features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = features // self.num_heads

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            use_bias=False,
        )

        query, key, value = (dense(name='query')(x),
                             dense(name='key')(x),
                             dense(name='value')(x))

        cos, sin = precompute_freqs_cis(dim=head_dim, end=seq_len, dtype=self.dtype)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        x = dot_product_attention(query, key, value, mask=mask, dtype=self.dtype)

        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='out',  # type: ignore[call-arg]
        )(x)
        return out


class TransformerBlock(nn.Module):
    config: TransformerConfig
    scan: bool = False

    @nn.compact
    def __call__(self, x):
        inputs, attn_mask = x
        config = self.config

        assert inputs.ndim == 3
        x = RMSNorm(epsilon=config.rms_norm_eps, dtype=config.dtype, name="ln_1")(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            name='attn')(x, attn_mask)
        x = x + inputs

        y = RMSNorm(epsilon=config.rms_norm_eps, dtype=config.dtype, name="ln_2")(x)
        y = MlpBlock(config=config, name='mlp')(y)
        if self.scan:
            return (x + y, attn_mask), None
        else:    
            return x + y, attn_mask
    

class TransformerModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train):
        config = self.config

        x = nn.Embed(
            num_embeddings=config.vocab_size, features=config.hidden_size, dtype=config.dtype, param_dtype=config.param_dtype, name='wte')(inputs)

        if attn_mask is not None:
            casual_mask = nn.make_causal_mask(attn_mask, dtype=attn_mask.dtype)
            attn_mask = attn_mask[:, None, None, :]
            attn_mask = nn.combine_masks(casual_mask, attn_mask)
        if config.remat_scan:
            remat_scan_lengths = config.remat_scan_lengths()
            TransformerBlockStack = nn.remat_scan(TransformerBlock, lengths=remat_scan_lengths)
            x = TransformerBlockStack(config, name='hs')((x, attn_mask))[0]
        else:
            block_fn = TransformerBlock
            if config.remat:
                block_fn = nn.remat(block_fn)
            if config.scan:
                TransformerBlockStack = nn.scan(block_fn, length=config.n_layers, variable_axes={True: 0}, split_rngs={True: True})
                x = TransformerBlockStack(config, scan=True, name='hs')((x, attn_mask))[0][0]
            else:
                for i in range(config.n_layers):
                    x = block_fn(config, name=f'h_{i}')((x, attn_mask))[0]
        norm = nn.remat(RMSNorm) if config.remat else RMSNorm
        x = norm(epsilon=config.rms_norm_eps, dtype=config.dtype, name='ln_f')(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(config=config, name='transformer')(inputs=inputs, attn_mask=attn_mask, train=train)

        batch_size = inputs.shape[0]
        seq_len = (jnp.not_equal(inputs, config.pad_token_id).sum(-1) - 1)
        x = x[jnp.arange(batch_size), seq_len]

        x = nn.Dense(
            config.num_labels,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name='score')(x)
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
    return root