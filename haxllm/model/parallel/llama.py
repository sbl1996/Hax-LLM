import functools
from typing import Any, Callable, Optional

import numpy as np

import jax.numpy as jnp
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning

from haxllm.model.llama import TransformerConfig, load_config, config_hub, remap_state_dict, precompute_freqs_cis, apply_rotary_pos_emb
from haxllm.model.modules import Dtype, Array, PRNGKey, Shape, default_kernel_init, DenseGeneral, dot_product_attention, RMSNorm
from haxllm.model.parallel.modules import DenseGeneral, Dense, Embed, remat_scan


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


class TransformerBlock(nn.Module):
    config: TransformerConfig
    scan: bool = False

    @nn.compact
    def __call__(self, x):
        inputs, attn_mask = x
        config = self.config

        casual_mask = nn.make_causal_mask(attn_mask, dtype=attn_mask.dtype)
        attn_mask_ = attn_mask[:, None, None, :]
        attn_mask_ = nn.combine_masks(casual_mask, attn_mask_)

        x = RMSNorm(epsilon=config.rms_norm_eps, dtype=config.dtype, name="ln_1")(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            name='attn')(x, attn_mask_)
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
        if config.remat or config.remat_scan:
            embed_layer = nn.remat(Embed)
        x = embed_layer(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": (None, "Y")},
            name='wte'
        )(inputs)

        remat_scan_lengths = config.remat_scan_lengths()
        TransformerBlockStack = remat_scan(
            TransformerBlock, lengths=remat_scan_lengths,
            variable_axes={True: 0}, split_rngs={True: True},
            metadata_params1={nn.PARTITION_NAME: None}, metadata_params2={nn.PARTITION_NAME: None})
        x = TransformerBlockStack(config, name='hs')((x, attn_mask))[0]

        norm_layer = nn.remat(RMSNorm) if config.remat or config.remat_scan else RMSNorm
        x = norm_layer(epsilon=config.rms_norm_eps, dtype=config.dtype, name='ln_f')(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        assert config.remat_scan, 'always use remat_scan=True for parallel model'
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