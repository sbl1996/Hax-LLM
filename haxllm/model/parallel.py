from typing import Mapping, Optional, Tuple, Callable, Sequence, Union, Literal
import functools
import dataclasses

import jax.numpy as jnp

import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.core import lift
from flax.linen import partitioning as nn_partitioning, initializers
from flax.linen.attention import (
    Dtype,
    Array,
    PRNGKey,
    Shape,
)

from haxllm.model.modules import DenseGeneral
from haxllm.gconfig import get_gconfig, get_attention_impl
from haxllm.model.attention import decode_for_padding, tpu_flash_attention, make_apply_rope, init_decode_cache, dot_product_attention
from haxllm.model.quantize import QConfig
from haxllm.model.mixin import RoPEScalingConfig

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
        fn = lambda scope, c: scan_fn(
            wrapper, length=lengths[0], metadata_params=metadata_params
        )(scope, c)[0]
    else:

        @functools.partial(lift.remat, policy=policy, prevent_cse=False)
        def inner_loop(scope, carry):
            carry = lift_remat_scan(
                body_fn,
                lengths[1:],
                policy,
                variable_broadcast,
                variable_carry,
                variable_axes,
                split_rngs,
                metadata_params,
            )(scope, carry)
            return carry, ()

        fn = lambda scope, c: scan_fn(
            inner_loop, length=lengths[0], metadata_params=metadata_params
        )(scope, c)[0]
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
        lift_remat_scan,
        target,
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
            raise ValueError(
                "ShardModule must have `shard` attribute or `shard` config option"
            )
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
            param = super().param(name, init_fn, *init_args) # type: ignore

            # Sow this, to have the AxisMetadata available at initialization.
            self.sow( # type: ignore
                "params_axes",
                f"{name}_axes",
                nn_partitioning.AxisMetadata(axes),
                reduce_fn=nn_partitioning._param_with_axes_sow_reduce_fn,
            )
        else:
            param = super().param(name, init_fn, *init_args) # type: ignore
        return param


class DenseGeneral(ShardMixIn, DenseGeneral):
    pass


class Embed(ShardMixIn, nn.Embed):
    pass


ShardAxis = Optional[str]
ModuleClass = Callable[..., nn.Module]


class SelfAttention(ShardModule):
    num_heads: int
    max_len: int
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    sliding_window_size: Optional[int] = None
    dtype: Optional[Dtype] = None
    param_dtype: Optional[Dtype] = jnp.float32
    broadcast_dropout: bool = False
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    qkv_bias: bool = True
    out_bias: bool = True
    is_causal: bool = True
    decode: bool = False
    rope: bool = False
    scale: Optional[float] = None
    attn_logits_soft_cap: Optional[float] = None
    rope_theta: float = 10000.0
    rope_scaling: Optional[RoPEScalingConfig] = None
    padding_left: bool = False
    query_shard_axes: Tuple[ShardAxis, ShardAxis, ShardAxis] = ("X", "Y", None)
    kv_shard_axes: Optional[Tuple[ShardAxis, ShardAxis, ShardAxis]] = None
    kv_cache_shard_axes: Optional[Tuple[ShardAxis, ShardAxis, ShardAxis, ShardAxis]] = None
    out_shard_axes: Tuple[ShardAxis, ShardAxis, ShardAxis] = ("Y", None, "X")
    shard: bool = True
    shard_cache: bool = False
    qconfig: Optional[QConfig] = None
    dense_cls: Union[ModuleClass, Sequence[ModuleClass]] = DenseGeneral

    @nn.compact
    def __call__(
        self,
        x: Array,
        padding_mask: Optional[Array] = None,
    ):
        r"""
        Parameters
        ----------
        x: Array, shape [batch, q_len, features]
            Input features.
        padding_mask: Optional[Array], shape [batch, q_len]
            Mask to indicate which query elements are padding.
            If both mask and padding_mask are provided, you must combine them by yourself.
            We only use padding_mask to infer position_ids.
        
        Returns
        -------
        out: Array, shape [batch, q_len, features]
            Output features.

        Notes
        -----
        In training and eval mode, either causal mask or padding mask is needed.
        In decode, only causal mask is supported and we infer it from cache.
        Therefor, mask is not needed for decoding.
        """
        get_qconfig = lambda name: self.qconfig if self.qconfig is not None and name in self.qconfig.q_layers else None

        multi_query = self.num_kv_heads is not None
        kv_shard_axes = self.kv_shard_axes or self.query_shard_axes

        features = x.shape[-1]
        assert (
            features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = self.head_dim or features // self.num_heads

        if not isinstance(self.dense_cls, Sequence):
            dense_cls = [self.dense_cls for _ in range(4)]
        else:
            assert len(self.dense_cls) == 4, "dense_cls must be a sequence of length 4 for query, key, value, and out."
            dense_cls = self.dense_cls

        query = dense_cls[0](
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.qkv_bias,
            shard_axes={"kernel": self.query_shard_axes},
            shard=self.shard,
            qconfig=get_qconfig("attn.query"),
            axis=-1,
            name="query",
        )(x)

        num_kv_heads = self.num_heads
        if multi_query:
            num_kv_heads = self.num_kv_heads
            kv_dense_shard_axes = None
        else:
            kv_dense_shard_axes = {"kernel": kv_shard_axes}

        kv_dense = [
            functools.partial(
                cls,
                features=(num_kv_heads, head_dim),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.qkv_bias,
                shard_axes=kv_dense_shard_axes,
                shard=self.shard,
                axis=-1,
            ) for cls in dense_cls[1:3]
        ]

        key = kv_dense[0](
            qconfig=get_qconfig("attn.key"), name="key")(x)
        value = kv_dense[1](
            qconfig=get_qconfig("attn.value"), name="value")(x)

        if self.rope:
            add_pos = make_apply_rope(head_dim, self.max_len, self.dtype, self.rope_theta, self.rope_scaling)
        else:
            add_pos = lambda q, k, p=None: (q, k)

        if not self.decode:
            if self.padding_left:
                raise NotImplementedError("padding_left=True is not supported for non-decode mode.")
            position_ids = None
            B = jnp.arange(key.shape[1])[None, :]
            idx = jnp.arange(query.shape[1])
            mask = B <= idx[:, None]
            if self.sliding_window_size is not None:
                mask = mask & (B > (idx - self.sliding_window_size)[:, None])
            query, key = add_pos(query, key, position_ids)
        else:
            kv_cache_shard_axes = self.kv_cache_shard_axes or (key.ndim - 2) * (None,) + kv_shard_axes[-2:]
            is_initialized, cached_key, cached_value, cache_index = init_decode_cache(self, key, value, kv_cache_shard_axes)
            if self.padding_left:
                cache_position = self.variable(
                    "cache", "cache_position", lambda: jnp.zeros(key.shape[0], dtype=jnp.int32)
                )
            else:
                cache_position = None

            mask = None
            if is_initialized:
                query, key, value, mask = decode_for_padding(
                    add_pos, query, key, value, cache_index, cached_key, cached_value,
                    self.padding_left, cache_position, padding_mask, self.sliding_window_size)

        dropout_rng = None
        if self.dropout_rate > 0 and not self.deterministic:
            dropout_rng = self.make_rng("dropout")
            deterministic = False
        else:
            deterministic = True

        if not self.decode and get_attention_impl() == 'flash':
            assert deterministic, "dropout not supported for flash attention."
            x = tpu_flash_attention(
                query,
                key,
                value,
                sliding_window_size=self.sliding_window_size,
                is_causal=self.is_causal,
                scale=self.scale,
                attn_logits_soft_cap=self.attn_logits_soft_cap,
                dtype=self.dtype,
                mesh=get_gconfig("mesh"),
            )
        else:
            x = dot_product_attention(
                query,
                key,
                value,
                mask=mask,
                scale=self.scale,
                attn_logits_soft_cap=self.attn_logits_soft_cap,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=deterministic,
                dtype=self.dtype,
            )

        out = dense_cls[3](
            features=features,
            axis=(-2, -1),
            use_bias=self.out_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            shard_axes={"kernel": self.out_shard_axes},
            shard=self.shard,
            qconfig=get_qconfig("attn.out"),
            name="out",
        )(x)
        return out


class MlpBlock(ShardModule):
    intermediate_size: Optional[int] = None
    activation: str = "gelu"
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    shard_axes1: Tuple[str, str] = ("X", "Y")
    shard_axes2: Tuple[str, str] = ("Y", "X")
    shard: bool = False
    dense_cls: Union[ModuleClass, Sequence[ModuleClass]] = DenseGeneral

    @nn.compact
    def __call__(self, inputs):
        assert self.activation in ["gelu", "gelu_new"]
        intermediate_size = self.intermediate_size or 4 * inputs.shape[-1]

        if not isinstance(self.dense_cls, Sequence):
            dense_cls = [self.dense_cls for _ in range(2)]
        else:
            assert len(self.dense_cls) == 2, "dense_cls must be a sequence of length 2 for fc_1 and fc_2"
            dense_cls = self.dense_cls

        dense = [
            functools.partial(
                cls,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                shard=self.shard,
            ) for cls in dense_cls
        ]

        actual_out_dim = inputs.shape[-1]
        x = dense[0](
            features=intermediate_size,
            shard_axes={"kernel": self.shard_axes1},
            name="fc_1",
        )(inputs)
        # x = self.with_sharding_constraint(
        #     x, ("X", None, "Y"))
        if self.activation == "gelu":
            x = nn.gelu(x, approximate=False)
        elif self.activation == "gelu_new":
            x = nn.gelu(x, approximate=True)
        x = dense[1](
            features=actual_out_dim,
            shard_axes={"kernel": self.shard_axes2},
            name="fc_2",
        )(x)
        return x


class GLUMlpBlock(ShardModule):
    intermediate_size: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    use_bias: bool = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    shard_axes1: Tuple[str, str] = ("X", "Y")
    shard_axes2: Tuple[str, str] = ("Y", "X")
    shard: bool = False
    qconfig: Optional[QConfig] = None
    dense_cls: Union[ModuleClass, Sequence[ModuleClass]] = DenseGeneral
    activation: Literal["swish", "gelu"] = "swish"

    @nn.compact
    def __call__(self, inputs):
        get_qconfig = lambda name: self.qconfig if self.qconfig is not None and name in self.qconfig.q_layers else None

        if not isinstance(self.dense_cls, Sequence):
            dense_cls = [self.dense_cls for _ in range(3)]
        else:
            assert len(self.dense_cls) == 3, "dense_cls must be a sequence of length 3 for gate, up and down"
            dense_cls = self.dense_cls

        dense = [
            functools.partial(
                # p_remat(DenseGeneral),  # Hack: remat here with less memory and same speed
                cls,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                shard=self.shard,
            ) for cls in dense_cls
        ]

        actual_out_dim = inputs.shape[-1]
        g = dense[0](
            features=self.intermediate_size,
            shard_axes={"kernel": self.shard_axes1},
            qconfig=get_qconfig("mlp.gate"),
            name="gate",
        )(inputs)
        if self.activation == "gelu":
            g = nn.gelu(g)
        elif self.activation == "swish":
            g = nn.swish(g)
        else:
            raise NotImplementedError
        x = g * dense[1](
            features=self.intermediate_size,
            shard_axes={"kernel": self.shard_axes1},
            qconfig=get_qconfig("mlp.up"),
            name="up",
        )(inputs)
        x = dense[2](
            features=actual_out_dim,
            shard_axes={"kernel": self.shard_axes2},
            qconfig=get_qconfig("mlp.down"),
            name="down",
        )(x)
        return x
