from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
from functools import partial

import math

import jax
import jax.numpy as jnp
from jax import lax

from flax.core import meta
import flax.linen as nn
from flax import struct
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype

from haxllm.model.parallel import ShardMixIn, DenseGeneral
from haxllm.model.quantize import QConfig, QuantMethod

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


default_kernel_init = initializers.kaiming_uniform()


@struct.dataclass
class LoraConfig:
    attn_lora_r: Tuple[int, int, int, int] = (8, 8, 0, 0)
    mlp_lora_r: Tuple[int, int, int] = (0, 0, 0)
    lora_alpha: int = 1

    def create_dense_cls(self, name: str):
        if name == "attn":
            rs = self.attn_lora_r
        elif name == "mlp":
            rs = self.mlp_lora_r
        else:
            raise ValueError(f"Unknown name: {name}")
        return [
            partial(LoraDenseGeneral, r=r, lora_alpha=self.lora_alpha)
            if r > 0 else DenseGeneral for r in rs
        ]


def _normalize_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


def _canonicalize_tuple(x: Union[Sequence[int], int]) -> Tuple[int, ...]:
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


class DenseGeneralBase(nn.Module):
    # TODO: lora_dropout
    features: Union[int, Sequence[int]]
    axis: Union[int, Sequence[int]] = -1
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    qconfig: Optional[QConfig] = None
    lora_param_dtype: Dtype = jnp.float32
    r: int = 0
    lora_alpha: int = 1

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        # TODO: support quantization
        if self.qconfig is not None:
            raise NotImplementedError("LoraDenseGeneral does not support quantization")
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)
        ndim = inputs.ndim
        axis = _normalize_axes(axis, ndim)
        n_axis, n_features = len(axis), len(features)

        def kernel_init_wrap(
            rng, shape, initializer, n_axis, n_features, dtype=jnp.float32
        ):
            flat_shape = (
                math.prod(shape[0:n_axis]),
                math.prod(shape[-n_features:]),
            )
            flat_shape = jax.tree.map(int, flat_shape)
            kernel = initializer(rng, flat_shape, dtype)
            if isinstance(kernel, meta.AxisMetadata):
                return meta.replace_boxed(kernel, jnp.reshape(kernel.unbox(), shape))
            return jnp.reshape(kernel, shape)

        expanded_batch_shape = tuple(
            1 for ax in range(inputs.ndim) if ax not in axis
        )
        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
        kernel = self.param(
            "kernel",
            kernel_init_wrap,
            kernel_shape,
            self.kernel_init,
            n_axis,
            n_features,
            self.param_dtype,
        )

        contract_ind = tuple(range(0, n_axis))

        if self.use_bias:
            def bias_init_wrap(rng, shape, dtype=jnp.float32):
                flat_shape = (math.prod(shape[-n_features:]),)
                bias = self.bias_init(rng, flat_shape, dtype)
                if isinstance(bias, meta.AxisMetadata):
                    return meta.replace_boxed(bias, jnp.reshape(bias.unbox(), shape))
                return jnp.reshape(bias, shape)

            bias = self.param("bias", bias_init_wrap, features, self.param_dtype)
        else:
            bias = None

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        out = lax.dot_general(
            inputs,
            kernel,
            ((axis, contract_ind), ((), ())),
        )
        if self.use_bias:
            bias = jnp.reshape(bias, expanded_batch_shape + features)
            out += bias

        if self.r > 0:
            scaling = self.lora_alpha / self.r

            lora_A_kernel_shape = tuple(inputs.shape[ax] for ax in axis) + (self.r,)
            lora_A_kernel = self.param(
                "lora_A_kernel",
                kernel_init_wrap,
                lora_A_kernel_shape,
                self.kernel_init,
                n_axis,
                1,
                self.lora_param_dtype,
            )
            inputs, lora_A_kernel = promote_dtype(inputs, lora_A_kernel, dtype=self.dtype)
            lora_A_out = lax.dot_general(
                inputs,
                lora_A_kernel,
                ((axis, contract_ind), ((), ())),
            )

            lora_B_kernel_shape = (self.r,) + features
            lora_B_kernel = self.param(
                "lora_B_kernel",
                kernel_init_wrap,
                lora_B_kernel_shape,
                initializers.zeros_init(),
                1,
                n_features,
                self.lora_param_dtype,
            )
            axis = (len(lora_A_out.shape) - 1,)
            contract_ind = (0,)
            lora_A_out, lora_B_kernel = promote_dtype(lora_A_out, lora_B_kernel, dtype=self.dtype)
            lora_B_out = scaling * lax.dot_general(
                lora_A_out,
                lora_B_kernel,
                ((axis, contract_ind), ((), ())),
            )

            out += lora_B_out
        return out


class LoraDenseGeneral(ShardMixIn, DenseGeneralBase):
    pass
