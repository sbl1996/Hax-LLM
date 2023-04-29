from typing import Any, Mapping, Optional, Tuple
import functools
import dataclasses

import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.core import lift
from flax.linen import partitioning as nn_partitioning


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


class Embed(ShardMixIn, nn.Embed):
    pass
