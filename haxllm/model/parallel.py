from typing import Mapping, Optional, Tuple, Callable
import functools
import dataclasses

import jax.numpy as jnp
from jax import lax

import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.core import lift
from flax.linen import partitioning as nn_partitioning, initializers
from flax.linen.attention import dot_product_attention, Dtype, Array, combine_masks, PRNGKey, Shape

from haxllm.gconfig import get as get_gconfig

default_kernel_init = initializers.lecun_normal()

def lift_remat_scan(
    body_fn,
    lengths,
    policy=None,
    variable_broadcast=False,
    variable_carry=False,
    variable_axes={True: 0},
    split_rngs={True: True},
    metadata_params={},
):
    scan_fn = functools.partial(
      lift.scan,
      variable_broadcast=variable_broadcast,
      variable_carry=variable_carry,
      variable_axes=variable_axes,
      split_rngs=split_rngs,
    )
    if len(lengths) == 1:
        # @functools.partial(lift.remat, policy=policy, prevent_cse=False)
        def wrapper(scope, carry):
            return body_fn(scope, carry), ()
        if get_gconfig("remat_scan_level") == 2:
            wrapper = lift.remat(wrapper, policy=policy, prevent_cse=False)
        fn = lambda scope, c: scan_fn(wrapper, length=lengths[0], metadata_params=metadata_params)(scope, c)[0]
    else:
        @functools.partial(lift.remat, policy=policy, prevent_cse=False)
        def inner_loop(scope, carry):
            carry = lift_remat_scan(body_fn, lengths[1:], policy,
                         variable_broadcast, variable_carry,
                         variable_axes, split_rngs, metadata_params)(scope, carry)
            return carry, ()
        fn = lambda scope, c: scan_fn(inner_loop, length=lengths[0], metadata_params=metadata_params)(scope, c)[0]
    return fn


def remat_scan(
    target,
    lengths=(),
    policy=None,
    variable_broadcast=False,
    variable_carry=False,
    variable_axes=FrozenDict({True: 0}),
    split_rngs=FrozenDict({True: True}),
    metadata_params={},
):
    return nn.transforms.lift_transform(
        lift_remat_scan, target,
        lengths=lengths,
        variable_broadcast=variable_broadcast,
        variable_carry=variable_carry,
        variable_axes=variable_axes,
        split_rngs=split_rngs,
        metadata_params=metadata_params,
        policy=policy,
    )

remat = nn_partitioning.remat

class ShardModule(nn.Module):

    def with_sharding_constraint(self, x, axes):
        try:
            shard = getattr(self, "shard", None)
            if shard is None:
               shard = self.config.shard
        except AttributeError:
            raise ValueError("ShardModule must have `shard` attribute or `shard` config option")
        if shard:
            return nn_partitioning.with_sharding_constraint(x, axes)  # type: ignore
        else:
            return x


@dataclasses.dataclass
class ShardMixIn:
    """Adds parameter sharding constraints for any flax.linen Module.
    This is a mix-in class that overrides the `param` method of the
    original Module, to selectively add sharding constraints as specified
    in `shard_axes`"""

    shard_axes: Optional[Mapping[str, Tuple[str, ...]]] = None
    shard: bool = True

    def param(self, name: str, init_fn, *init_args):
        # If `shard_axes` specified and param name in the dict, apply constraint
        if self.shard and self.shard_axes and (name in self.shard_axes.keys()):
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


class Embed(ShardMixIn, nn.Embed):
    pass

ShardAxis = Optional[str]

class SelfAttention(ShardModule):
    num_heads: int
    max_len: int
    dtype: Optional[Dtype] = None
    param_dtype: Optional[Dtype] = jnp.float32
    broadcast_dropout: bool = False
    dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    use_bias: bool = True
    decode: bool = False
    qkv_shard_axes: Tuple[ShardAxis, ShardAxis, ShardAxis] = ("X", None, "Y")
    out_shard_axes: Tuple[ShardAxis, ShardAxis, ShardAxis] = ("Y", None, "X")
    shard: bool = True

    @nn.compact
    def __call__(self, x, mask: Optional[Array] = None):
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
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            shard_axes={"kernel": self.qkv_shard_axes},
            shard=self.shard,
        )

        qkv_constraint = lambda x: x
        # qkv_constraint = functools.partial(
        #     self.with_sharding_constraint, axes=("X", None, "Y", None))

        query, key, value = (
            qkv_constraint(dense(name='query')(x)),
            qkv_constraint(dense(name='key')(x)),
            qkv_constraint(dense(name='value')(x)),
        )

        if self.decode:
            is_initialized = self.has_variable('cache', 'cached_key')
            zero_init = jnp.zeros
            if self.shard:
                zero_init = nn.with_partitioning(zero_init, self.qkv_shard_axes)
            cached_key = self.variable('cache', 'cached_key',
                                       zero_init, key.shape, key.dtype)
            cached_value = self.variable('cache', 'cached_value',
                                         zero_init, value.shape, value.dtype)
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
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                                     tuple(batch_dims) + (1, 1, max_length)))

        dropout_rng = None
        if self.dropout_rate > 0 and not self.deterministic:
            dropout_rng = self.make_rng('dropout')
            deterministic = False
        else:
            deterministic = True


        x = dot_product_attention(
            query, key, value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype)

        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            shard_axes={"kernel": self.out_shard_axes},
            shard=self.shard,
            name='out',
        )(x)  # type: ignore
        # out = self.with_sharding_constraint(
        #     out, ("X", None, "Y"))
        return out


class MlpBlock(ShardModule):
    intermediate_size: Optional[int] = None
    activation: str = 'gelu'
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    shard_axes1: Tuple[str, str] = ("X", "Y")
    shard_axes2: Tuple[str, str] = ("Y", "X")
    shard: bool = False

    @nn.compact
    def __call__(self, inputs):
        assert self.activation in ['gelu', 'gelu_new']
        intermediate_size = self.intermediate_size or 4 * inputs.shape[-1]

        dense = functools.partial(
            Dense,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            shard=self.shard,
        )

        actual_out_dim = inputs.shape[-1]
        x = dense(
            features=intermediate_size,
            shard_axes={"kernel": self.shard_axes1},
            name="fc_1")(inputs)
        # x = self.with_sharding_constraint(
        #     x, ("X", None, "Y"))
        if self.activation == 'gelu':
            x = nn.gelu(x, approximate=False)
        elif self.activation == 'gelu_new':
            x = nn.gelu(x, approximate=True)
        x = dense(
            features=actual_out_dim,
            shard_axes={"kernel": self.shard_axes2},
            name="fc_2")(x)
        return x
