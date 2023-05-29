import functools
from typing import Optional

import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention, Dtype, Array, combine_masks


from haxllm.model.gpt2 import TransformerConfig, load_config
from haxllm.model.parallel.modules import DenseGeneral, Dense, Embed, remat_scan


class SelfAttention(nn.Module):
    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Optional[Dtype] = None
    dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    use_bias: bool = True
    decode: bool = False

    @nn.compact
    def __call__(self, x, mask: Optional[Array] = None):
        features = x.shape[-1]
        assert features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = features // self.num_heads

        qkv_dense = functools.partial(
            DenseGeneral,
            axis=-1,
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.use_bias,
            shard_axes={"kernel": ("X", "Y", None)},
        )

        qkv_constraint = functools.partial(
            nn_partitioning.with_sharding_constraint,
            logical_axis_resources=("X", None, "Y", None),
        )

        query, key, value = (
            qkv_constraint(qkv_dense(name='query')(x)),
            qkv_constraint(qkv_dense(name='key')(x)),
            qkv_constraint(qkv_dense(name='value')(x)),
        )

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.decode:
            # detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable('cache', 'cached_key')
            cached_key = self.variable('cache', 'cached_key',
                                       jnp.zeros, key.shape, key.dtype)
            cached_value = self.variable('cache', 'cached_value',
                                         jnp.zeros, value.shape, value.dtype)
            cache_index = self.variable('cache', 'cache_index',
                                        lambda: jnp.array(0, dtype=jnp.int32))
            if is_initialized:
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
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(
                    cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # causal mask for cached decoder self-attention:
                # our single query position should only attend to those key
                # positions that have already been generated and cached,
                # not the remaining zero elements.
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                                     tuple(batch_dims) + (1, 1, max_length)))

        dropout_rng = None
        if self.dropout_rate > 0 and not self.deterministic:
            dropout_rng = self.make_rng('dropout')

        x = dot_product_attention(
            query, key, value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=False,
            deterministic=self.deterministic,
            dtype=self.dtype)
        x = nn_partitioning.with_sharding_constraint(
            x, ("X", None, "Y", None))

        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            use_bias=self.use_bias,
            dtype=self.dtype,
            shard_axes={"kernel": ("Y", None, "X")},
            name='out',
        )(x)
        out = nn_partitioning.with_sharding_constraint(
            out, ("X", None, "Y"))
        return out


class MlpBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        intermediate_size = config.hidden_size * 4

        actual_out_dim = inputs.shape[-1]
        x = Dense(
            intermediate_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"kernel": ("X", "Y")},
            name="fc_1")(inputs)
        x = nn_partitioning.with_sharding_constraint(
            x, ("X", None, "Y"))
        x = nn.gelu(x)
        x = Dense(
            actual_out_dim,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"kernel": ("Y", "X")},
            name="fc_2")(x)
        x = nn.Dropout(rate=config.resid_pdrop)(x, deterministic=self.deterministic)
        return x


class TransformerBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        attn_mask = nn.make_causal_mask(inputs[..., 0], dtype=jnp.bool_)

        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name='ln_1')(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            dropout_rate=config.attn_pdrop,
            deterministic=self.deterministic,
            decode=config.decode,
            name='attn')(x, attn_mask)
        x = x + inputs

        y = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name='ln_2')(x)
        y = MlpBlock(config=config, deterministic=self.deterministic, name='mlp')(y)
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
        )

        position_ids = jnp.arange(0, inputs.shape[-1], dtype=jnp.int32)[None]
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

        if config.remat_scan:
            remat_scan_lengths = config.remat_scan_lengths()
            TransformerBlockStack = remat_scan(
                TransformerBlock, lengths=remat_scan_lengths,
                variable_axes={True: 0}, split_rngs={True: True},
                metadata_params1={nn.PARTITION_NAME: None}, metadata_params2={nn.PARTITION_NAME: None})
            x = TransformerBlockStack(
                config, deterministic=not train, name='hs')(x)
        else:
            if config.remat and train:
                block_fn = nn.remat(TransformerBlock)
            else:
                block_fn = TransformerBlock
            TransformerBlockStack = nn.scan(
                block_fn, length=config.scan_layers(), variable_axes={True: 0},
                split_rngs={True: True}, metadata_params={nn.PARTITION_NAME: None})
            x = TransformerBlockStack(config, deterministic=not train, scan=True, name='hs')(x)[0]

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