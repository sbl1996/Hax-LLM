import math
import functools
from typing import Optional

import jax.numpy as jnp
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention, Dtype, Array, merge_param


from haxllm.model.gpt2 import TransformerConfig, convert_config, remap_state_dict
from haxllm.model.parallel.modules import DenseGeneral, Dense, Embed, remat_scan


class MultiHeadDotProductAttention(nn.Module):
    num_heads: int
    dtype: Optional[Dtype] = None
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    use_bias: bool = True

    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 mask: Optional[Array] = None,
                 deterministic: Optional[bool] = None):

        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

        qkv_dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            features=(self.num_heads, head_dim),
            use_bias=self.use_bias,
            shard_axes={"kernel": ("X", "Y", None)},
        )

        qkv_constraint = functools.partial(
            nn_partitioning.with_sharding_constraint,
            logical_axis_resources=("X", None, "Y", None),
        )

        query, key, value = (
            qkv_constraint(qkv_dense(name='query')(inputs_q)),
            qkv_constraint(qkv_dense(name='key')(inputs_kv)),
            qkv_constraint(qkv_dense(name='value')(inputs_kv)),
        )

        dropout_rng = None
        if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
            m_deterministic = merge_param('deterministic', self.deterministic,
                                          deterministic)
            if not m_deterministic:
                dropout_rng = self.make_rng('dropout')
        else:
            m_deterministic = True

        x = dot_product_attention(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=m_deterministic,
            dtype=self.dtype)  # pytype: disable=wrong-keyword-args
        x = nn_partitioning.with_sharding_constraint(
            x, ("X", None, "Y", None))

        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            use_bias=self.use_bias,
            dtype=self.dtype,
            shard_axes={"kernel": ("Y", None, "X")},
            name='out',  # type: ignore[call-arg]
        )(x)
        out = nn_partitioning.with_sharding_constraint(
            out, ("X", None, "Y"))
        return out


class SelfAttention(MultiHeadDotProductAttention):

    @nn.compact
    def __call__(self, inputs_q: Array, mask: Optional[Array] = None,  # type: ignore
                 deterministic: Optional[bool] = None):

        return super().__call__(inputs_q, inputs_q, mask,
                                deterministic=deterministic)


class MlpBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, deterministic=True):
        config = self.config
        intermediate_size = config.hidden_size * 4

        actual_out_dim = inputs.shape[-1]
        x = Dense(
            intermediate_size,
            dtype=config.dtype,
            shard_axes={"kernel": ("X", "Y")},
            name="fc_1")(inputs)
        x = nn_partitioning.with_sharding_constraint(
            x, ("X", None, "Y"))
        x = nn.gelu(x)
        x = Dense(
            actual_out_dim,
            dtype=config.dtype,
            shard_axes={"kernel": ("Y", "X")},
            name="fc_2")(x)
        x = nn.Dropout(rate=config.resid_pdrop)(x, deterministic=deterministic)
        return x


class TransformerBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool

    @nn.compact
    def __call__(self, x):
        inputs, attn_mask = x
        config = self.config
        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name='ln_1')(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            dtype=config.dtype,
            qkv_features=config.hidden_size,
            use_bias=True,
            broadcast_dropout=False,
            dropout_rate=config.attn_pdrop,
            deterministic=self.deterministic,
            name='attn')(x, attn_mask)
        x = x + inputs

        y = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name='ln_2')(x)
        y = MlpBlock(config=config, name='mlp')(
            y, deterministic=self.deterministic)
        # return (x + y, attn_mask), None
        return x + y, attn_mask


class TransformerModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train):
        config = self.config

        position_ids = jnp.arange(0, inputs.shape[-1], dtype=jnp.int32)[None]

        inputs_embeds = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            shard_axes={"embedding": (None, "Y")},
            name='wte'
        )(inputs)
        position_embeds = nn.Embed(
            num_embeddings=config.n_positions, features=config.hidden_size, dtype=config.dtype, name='wpe')(position_ids)

        x = inputs_embeds + position_embeds

        x = nn.Dropout(rate=config.embd_pdrop)(x, deterministic=not train)

        if attn_mask is not None:
            if config.is_casual:
                casual_mask = nn.make_causal_mask(
                    attn_mask, dtype=attn_mask.dtype)
                attn_mask = jnp.expand_dims(attn_mask, (-2, -3))
                attn_mask = nn.combine_masks(casual_mask, attn_mask)
            else:
                attn_mask = jnp.expand_dims(attn_mask, (-2, -3))

        if config.remat_scan_lengths is not None:
            remat_scan_layers = math.prod(config.remat_scan_lengths)
            # TransformerBlockStack = nn.scan(
            #     nn.remat(TransformerBlock), length=remat_scan_layers, 
            #     variable_axes={"params": 0}, split_rngs={True: True},
            #     metadata_params={nn.PARTITION_NAME: 'layer'}
            # )
            # x = TransformerBlockStack(
            #     config, deterministic=not train, name='hs')((x, attn_mask))[0][0]
            d = config.n_layers - remat_scan_layers
            if d < 0:
                raise ValueError(
                    f"remat_scan_lengths={config.remat_scan_lengths} is too large for num_hidden_layers={config.n_layers}")
            for i in range(d):
                x = TransformerBlock(config, deterministic=not train, name=f'h_{i}')((x, attn_mask))[0]
            TransformerBlockStack = remat_scan(
                TransformerBlock, lengths=config.remat_scan_lengths,
                variable_axes={"params": 0}, split_rngs={True: True},
                metadata_params1={nn.PARTITION_NAME: 'layer1'}, metadata_params2={nn.PARTITION_NAME: 'layer2'})
            x = TransformerBlockStack(
                config, deterministic=not train, name='hs')((x, attn_mask))[0]
        else:
            if config.remat and train:
                block_fn = nn.remat(TransformerBlock)
            else:
                block_fn = TransformerBlock
            for i in range(config.n_layers):
                x = block_fn(config, deterministic=not train, name=f'h_{i}')((x, attn_mask))[0]
        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name='ln_f')(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(config=config, name='transformer')(
            inputs=inputs, attn_mask=attn_mask, train=train)

        batch_size = inputs.shape[0]
        seq_len = (jnp.not_equal(inputs, config.pad_token_id).sum(-1) - 1)
        x = x[jnp.arange(batch_size), seq_len]

        x = nn.Dense(
            config.num_labels,
            dtype=config.dtype,
            name='score')(x)
        return x
