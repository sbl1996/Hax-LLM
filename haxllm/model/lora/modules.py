import functools
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import math

import jax
import jax.numpy as jnp
from jax import lax

from flax.core import meta
import flax.linen as nn
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.attention import combine_masks, merge_param, dot_product_attention

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


default_kernel_init = initializers.kaiming_uniform()


def _normalize_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


def _canonicalize_tuple(x: Union[Sequence[int], int]) -> Tuple[int, ...]:
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


class DenseGeneral(nn.Module):
    # TODO: lora_dropout
    features: Union[int, Sequence[int]]
    axis: Union[int, Sequence[int]] = -1
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    lora_param_dtype: Dtype = jnp.float32
    r: int = 0
    lora_alpha: int = 1

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        kernel_init = default_kernel_init
        bias_init = initializers.zeros_init()

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
            kernel = initializer(rng, flat_shape, dtype)
            if isinstance(kernel, meta.AxisMetadata):
                return meta.replace_boxed(kernel, jnp.reshape(kernel.unbox(), shape))
            return jnp.reshape(kernel, shape)

        expanded_batch_shape = tuple(1 for ax in range(inputs.ndim) if ax not in axis)
        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features

        kernel = self.param(
            "kernel",
            kernel_init_wrap,
            kernel_shape,
            kernel_init,
            n_axis,
            n_features,
            self.param_dtype,
        )

        if self.use_bias:

            def bias_init_wrap(rng, shape, dtype=jnp.float32):
                flat_shape = (math.prod(shape[-n_features:]),)
                bias = bias_init(rng, flat_shape, dtype)
                if isinstance(bias, meta.AxisMetadata):
                    return meta.replace_boxed(bias, jnp.reshape(bias.unbox(), shape))
                return jnp.reshape(bias, shape)

            bias = self.param("bias", bias_init_wrap, features, self.param_dtype)
        else:
            bias = None

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        contract_ind = tuple(range(0, n_axis))
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
                kernel_init,
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


class MultiHeadDotProductAttention(nn.Module):
    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    use_bias: bool = True
    decode: bool = False
    attn_lora_r: Sequence[int] = (0, 0, 0, 0)
    lora_alpha: int = 1

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array] = None,
        deterministic: Optional[bool] = None,
    ):
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert (
            qkv_features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = qkv_features // self.num_heads

        assert len(self.attn_lora_r) == 4, "attn_lora_r must be of length 4 for query, key, value, and out."

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            use_bias=self.use_bias,
            lora_alpha=self.lora_alpha,
        )

        query, key, value = (
            dense(r=self.attn_lora_r[0], name="query")(inputs_q),
            dense(r=self.attn_lora_r[1], name="key")(inputs_kv),
            dense(r=self.attn_lora_r[2], name="value")(inputs_kv),
        )

        # TODO: decode
        # # During fast autoregressive decoding, we feed one position at a time,
        # # and cache the keys and values step by step.
        # if self.decode:
        #     # detect if we're initializing by absence of existing cache data.
        #     is_initialized = self.has_variable("cache", "cached_key")
        #     cached_key = self.variable(
        #         "cache", "cached_key", jnp.zeros, key.shape, key.dtype
        #     )
        #     cached_value = self.variable(
        #         "cache", "cached_value", jnp.zeros, value.shape, value.dtype
        #     )
        #     cache_index = self.variable(
        #         "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
        #     )
        #     if is_initialized:
        #         (
        #             *batch_dims,
        #             max_length,
        #             num_heads,
        #             depth_per_head,
        #         ) = cached_key.value.shape
        #         # shape check of cached keys against query input
        #         expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        #         if expected_shape != query.shape:
        #             raise ValueError(
        #                 "Autoregressive cache shape error, "
        #                 "expected query shape %s instead got %s."
        #                 % (expected_shape, query.shape)
        #             )
        #         # update key, value caches with our new 1d spatial slices
        #         cur_index = cache_index.value
        #         indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        #         key = lax.dynamic_update_slice(cached_key.value, key, indices)
        #         value = lax.dynamic_update_slice(cached_value.value, value, indices)
        #         cached_key.value = key
        #         cached_value.value = value
        #         cache_index.value = cache_index.value + 1
        #         # causal mask for cached decoder self-attention:
        #         # our single query position should only attend to those key
        #         # positions that have already been generated and cached,
        #         # not the remaining zero elements.
        #         mask = combine_masks(
        #             mask,
        #             jnp.broadcast_to(
        #                 jnp.arange(max_length) <= cur_index,
        #                 tuple(batch_dims) + (1, 1, max_length),
        #             ),
        #         )

        dropout_rng = None
        if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
            m_deterministic = merge_param(
                "deterministic", self.deterministic, deterministic
            )
            if not m_deterministic:
                dropout_rng = self.make_rng("dropout")
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
            dtype=self.dtype,
        )  # pytype: disable=wrong-keyword-args

        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            r=self.attn_lora_r[3],
            lora_alpha=self.lora_alpha,
            name="out",  # type: ignore[call-arg]
        )(x)
        return out


class SelfAttention(MultiHeadDotProductAttention):
    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        mask: Optional[Array] = None,  # type: ignore
        deterministic: Optional[bool] = None,
    ):
        return super().__call__(inputs_q, inputs_q, mask, deterministic=deterministic)
