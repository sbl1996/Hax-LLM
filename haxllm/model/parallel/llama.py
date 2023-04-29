import functools
import math
from typing import Callable, Optional

import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning


from haxllm.model.llama import TransformerConfig, convert_config, remap_state_dict
from haxllm.model.modules import Dtype, Array, PRNGKey, Shape, default_kernel_init, dot_product_attention, RMSNorm
from haxllm.model.parallel.modules import DenseGeneral, Dense, Embed, remat_scan


class MlpBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        dense = functools.partial(
            Dense,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
        )

        actual_out_dim = inputs.shape[-1]
        g = nn.silu(dense(
            features=config.intermediate_size,
            shard_axes={"kernel": ("X", "Y")},
            name="gate")(inputs))
        g = nn_partitioning.with_sharding_constraint(
            g, ("X", None, "Y"))
        x = g * dense(
            features=config.intermediate_size,
            shard_axes={"kernel": ("X", "Y")},
            name="up")(inputs)
        x = nn_partitioning.with_sharding_constraint(
            x, ("X", None, "Y"))
        x = dense(
            features=actual_out_dim,
            shard_axes={"kernel": ("Y", "X")},
            name="down")(x)
        return x


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
            shard_axes={"kernel": ("X", "Y", None)},
        )

        qkv_constraint = functools.partial(
            nn_partitioning.with_sharding_constraint,
            logical_axis_resources=("X", None, "Y", None),
        )

        query, key, value = (
            qkv_constraint(dense(name='query')(x)),
            qkv_constraint(dense(name='key')(x)),
            qkv_constraint(dense(name='value')(x)),
        )

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
            shard_axes={"kernel": ("Y", None, "X")},
            name='out',
        )(x)
        out = nn_partitioning.with_sharding_constraint(
            out, ("X", None, "Y"))
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

        embed_layer = Embed
        if config.remat_scan_lengths is not None or config.remat:
            embed_layer = nn.remat(Embed)
        x = embed_layer(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": (None, "Y")},
            name='wte'
        )(inputs)

        if attn_mask is not None:
            casual_mask = nn.make_causal_mask(attn_mask, dtype=attn_mask.dtype)
            attn_mask = attn_mask[:, None, None, :]
            attn_mask = nn.combine_masks(casual_mask, attn_mask)
        if config.remat_scan_lengths is not None:
            remat_scan_layers = math.prod(config.remat_scan_lengths)
            d = config.n_layers - remat_scan_layers
            if d < 0:
                raise ValueError(f"remat_scan_lengths={config.remat_scan_lengths} is too large for n_layers={config.n_layers}")
            for i in range(d):
                x = TransformerBlock(config, name=f'h_{i}')((x, attn_mask))[0]
            TransformerBlockStack = remat_scan(
                TransformerBlock, lengths=config.remat_scan_lengths,
                variable_axes={"params": 0}, split_rngs={True: True},
                metadata_params1={nn.PARTITION_NAME: None}, metadata_params2={nn.PARTITION_NAME: None})
            x = TransformerBlockStack(config, name='hs')((x, attn_mask))[0]
        else:
            block_fn = TransformerBlock
            if config.remat:
                block_fn = nn.remat(block_fn)
            scan_layers = config.scan_layers
            d = config.n_layers - scan_layers
            if d < 0:
                raise ValueError(f"scan_layers={config.scan_layers} is too large for n_layers={config.n_layers}")
            for i in range(d):
                x = block_fn(config, name=f'h_{i}')((x, attn_mask))[0]
            if scan_layers > 0:
                TransformerBlockStack = nn.scan(
                    block_fn, length=scan_layers, variable_axes={True: 0},
                    metadata_params={nn.PARTITION_NAME: None}, split_rngs={True: True})
                x = TransformerBlockStack(config, scan=True, name='hs')((x, attn_mask))[0][0]
        norm_layer = RMSNorm
        if config.remat_scan_lengths is not None or config.remat:
            norm_layer = nn.remat(RMSNorm)
        x = norm_layer(epsilon=config.rms_norm_eps, dtype=config.dtype, name='ln_f')(x)
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

        x = Dense(
            config.num_labels,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name='score')(x)
        return x