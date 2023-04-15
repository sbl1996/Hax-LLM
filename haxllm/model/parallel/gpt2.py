import math
import dataclasses
import functools
from typing import Any, Mapping, Optional, Tuple

import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.core import lift
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention, Dtype, Array, merge_param

nn_partitioning.remat
def lift_remat_scan(
    body_fn,
    lengths,
    policy=None,
    variable_broadcast=False,
    variable_carry=False,
    variable_axes={True: 0},
    split_rngs={True: True},
    metadata_params1={},
    metadata_params2={},
):
  scan_fn = functools.partial(
      lift.scan,
      variable_broadcast=variable_broadcast,
      variable_carry=variable_carry,
      variable_axes=variable_axes,
      split_rngs=split_rngs,
    #   metadata_params=metadata_params,
    )
  if len(lengths) == 1:
    def wrapper(scope, carry):
      return body_fn(scope, carry), ()
    fn = lambda scope, c: scan_fn(wrapper, length=lengths[0], metadata_params=metadata_params2)(scope, c)[0]
  else:
    @functools.partial(lift.remat, policy=policy, prevent_cse=False)
    def inner_loop(scope, carry):
      carry = lift_remat_scan(body_fn, lengths[1:], policy,
                         variable_broadcast, variable_carry,
                         variable_axes, split_rngs, metadata_params1, metadata_params2)(scope, carry)
      return carry, ()
    fn = lambda scope, c: scan_fn(inner_loop, length=lengths[0], metadata_params=metadata_params1)(scope, c)[0]
  return fn


def remat_scan(
    target,
    lengths=(),
    policy=None,
    variable_broadcast=False,
    variable_carry=False,
    variable_axes=FrozenDict({True: 0}),
    split_rngs=FrozenDict({True: True}),
    metadata_params1={},
    metadata_params2={},
):
    return nn.transforms.lift_transform(
        lift_remat_scan, target,
        lengths=lengths,
        variable_broadcast=variable_broadcast,
        variable_carry=variable_carry,
        variable_axes=variable_axes,
        split_rngs=split_rngs,
        metadata_params1=metadata_params1,
        metadata_params2=metadata_params2,
        policy=policy,
    )


def convert_config(config, **kwargs):
    d = {}
    for k in TransformerConfig.__annotations__.keys():
        if hasattr(config, k):
            v = getattr(config, k)
            if v is not None:
                d[k] = v
    for k, v in kwargs.items():
        d[k] = v
    return TransformerConfig(**d)


@struct.dataclass
class TransformerConfig:
    vocab_size: int = 50257
    num_labels: int = 2
    dtype: Any = jnp.float32
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    layer_norm_epsilon: float = 1e-5
    n_positions: int = 1024
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    pad_token_id: int = 50256
    is_casual: bool = True
    remat: bool = False
    remat_scan_lengths: Optional[Tuple[int, int]] = None


@dataclasses.dataclass
class ShardMixIn:
    """Adds parameter sharding constraints for any flax.linen Module.
    This is a mix-in class that overrides the `param` method of the
    original Module, to selectively add sharding constraints as specified
    in `shard_axes`"""

    shard_axes: Optional[Mapping[str, Tuple[str, ...]]] = None

    def param(self, name: str, init_fn, *init_args):
        # If `shard_axes` specified and param name in the dict, apply constraint
        if self.shard_axes and (name in self.shard_axes.keys()):
            axes = self.shard_axes[name]
            init_fn = nn.with_partitioning(init_fn, axes)
            param = super().param(name, init_fn, *init_args)

            # Sow this, to have the AxisMetadata available at initialization.
            self.sow(
                "params_axes",
                f"{name}_axes",
                nn_partitioning.AxisMetadata(axes),
                reduce_fn=nn_partitioning._param_with_axes_sow_reduce_fn,
            )
        else:
            param = super().param(name, init_fn, *init_args)
        return param


class DenseGeneral(ShardMixIn, nn.DenseGeneral):
    pass


class Dense(ShardMixIn, nn.Dense):
    pass


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
        n_inner = config.n_embd * 4

        actual_out_dim = inputs.shape[-1]
        x = Dense(
            n_inner,
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
        x = nn_partitioning.with_sharding_constraint(
            x, ("X", None, "Y"))
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
            num_heads=config.n_head,
            dtype=config.dtype,
            qkv_features=config.n_embd,
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

        inputs_embeds = nn.Embed(
            num_embeddings=config.vocab_size, features=config.n_embd, dtype=config.dtype, name='wte')(inputs)
        position_embeds = nn.Embed(
            num_embeddings=config.n_positions, features=config.n_embd, dtype=config.dtype, name='wpe')(position_ids)

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
            d = config.n_layer - remat_scan_layers
            if d < 0:
                raise ValueError(
                    f"remat_scan_lengths={config.remat_scan_lengths} is too large for n_layer={config.n_layer}")
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
            for i in range(config.n_layer):
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
