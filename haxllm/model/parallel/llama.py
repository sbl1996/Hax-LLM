import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax

import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention, Dtype, Array, combine_masks

from haxllm.model.llama import TransformerConfig, load_config, config_hub, remap_state_dict, precompute_freqs_cis, apply_rotary_pos_emb
from haxllm.model.modules import PRNGKey, Shape, default_kernel_init, DenseGeneral, RMSNorm
from haxllm.model.parallel.modules import DenseGeneral, Dense, Embed, remat_scan
from haxllm.model.memory_efficient_attention import dot_product_attention as dot_product_attention_m


class SelfAttention(nn.Module):
    num_heads: int
    max_len: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    decode: bool = False
    memory_efficient: bool = False

    @nn.compact
    def __call__(self, x: Array, mask: Optional[Array] = None):
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

        cos, sin = precompute_freqs_cis(dim=head_dim, end=self.max_len, dtype=self.dtype)

        if not self.decode:
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        else:
            is_initialized = self.has_variable('cache', 'cached_key')
            zero_init = nn.with_partitioning(jnp.zeros, (None, None, "Y", None))
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
    def __call__(self, *, inputs, train):
        config = self.config
        remat = config.remat or config.remat_scan

        embed_layer = nn.remat(Embed) if remat else Embed
        x = embed_layer(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": (None, "Y")},
            name='wte'
        )(inputs)

        if config.remat_scan:
            remat_scan_lengths = config.remat_scan_lengths()
            TransformerBlockStack = remat_scan(
                TransformerBlock, lengths=remat_scan_lengths,
                variable_axes={True: 0}, split_rngs={True: True},
                metadata_params1={nn.PARTITION_NAME: None}, metadata_params2={nn.PARTITION_NAME: None})
            x = TransformerBlockStack(config, name='hs')(x)
        else:
            block_fn = TransformerBlock
            if config.remat:
                block_fn = nn.remat(block_fn)
            if config.scan:
                TransformerBlockStack = nn.scan(
                    block_fn, length=config.scan_layers(), variable_axes={True: 0},
                    split_rngs={True: True}, metadata_params={nn.PARTITION_NAME: None})
                x = TransformerBlockStack(config, scan=True, name='hs')(x)[0]
            else:
                for i in range(config.n_layers):
                    x = block_fn(config, name=f'h_{i}')(x)

        norm_layer = nn.remat(RMSNorm) if remat else RMSNorm
        x = norm_layer(epsilon=config.rms_norm_eps, dtype=config.dtype, name='ln_f')(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        assert config.remat_scan, 'always use remat_scan=True for parallel model'
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