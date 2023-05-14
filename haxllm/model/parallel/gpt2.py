import functools
from typing import Optional

import jax.numpy as jnp
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention, Dtype, Array


from haxllm.model.gpt2 import TransformerConfig, load_config
from haxllm.model.parallel.modules import DenseGeneral, Dense, Embed, remat_scan


class SelfAttention(nn.Module):
    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Optional[Dtype] = None
    dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    use_bias: bool = True

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
    def __call__(self, x):
        inputs, attn_mask = x
        config = self.config

        casual_mask = nn.make_causal_mask(attn_mask, dtype=attn_mask.dtype)
        attn_mask_ = attn_mask[:, None, None, :]
        attn_mask_ = nn.combine_masks(casual_mask, attn_mask_)

        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name='ln_1')(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            dropout_rate=config.attn_pdrop,
            deterministic=self.deterministic,
            name='attn')(x, attn_mask_)
        x = x + inputs

        y = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name='ln_2')(x)
        y = MlpBlock(config=config, deterministic=self.deterministic, name='mlp')(y)
        if self.scan:
            return (x + y, attn_mask), None
        else:    
            return x + y, attn_mask


class TransformerModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train):
        config = self.config
        remat = config.remat or config.remat_scan

        embed_layer = Embed
        if remat:
            embed_layer = nn.remat(Embed)

        embed_layer = functools.partial(
            embed_layer,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": (None, "Y")},
        )

        position_ids = jnp.arange(0, inputs.shape[-1], dtype=jnp.int32)[None]

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
                config, deterministic=not train, name='hs')((x, attn_mask))[0]
        else:
            if config.remat and train:
                block_fn = nn.remat(TransformerBlock)
            else:
                block_fn = TransformerBlock
            TransformerBlockStack = nn.scan(
                block_fn, length=config.scan_layers(), variable_axes={True: 0},
                split_rngs={True: True}, metadata_params={nn.PARTITION_NAME: None})
            x = TransformerBlockStack(config, deterministic=not train, scan=True, name='hs')((x, attn_mask))[0][0]
        norm_layer = nn.remat(nn.LayerNorm) if remat else nn.LayerNorm
        x = norm_layer(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_f')(x)
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
