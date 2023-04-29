import math
import functools
from typing import Callable, Optional, Tuple, Union, Sequence, Iterable

import jax
import jax.numpy as jnp
from jax import lax

from flax.core import meta
import flax.linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen import Module, compact, initializers
from flax.linen.attention import Dtype, Array, PRNGKey, Shape, combine_masks, merge_param, dot_product_attention


default_kernel_init = initializers.lecun_normal()


def _normalize_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


def _canonicalize_tuple(x: Union[Sequence[int], int]) -> Tuple[int, ...]:
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


class RMSNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        dtype = jnp.promote_types(self.dtype, jnp.float32)
        x = jnp.asarray(x, dtype)
        x = x * jax.lax.rsqrt(jnp.square(x).mean(-1, keepdims=True) + self.epsilon)

        reduced_feature_shape = (x.shape[-1],)
        scale = self.param('scale', nn.initializers.ones, reduced_feature_shape, self.param_dtype)
        x = x * scale
        return jnp.asarray(x, self.dtype)


class DenseGeneral(nn.Module):
    features: Union[int, Sequence[int]]
    axis: Union[int, Sequence[int]] = -1
    batch_dims: Sequence[int] = ()
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs: Array) -> Array:

        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)
        ndim = inputs.ndim
        axis = _normalize_axes(axis, ndim)
        n_axis, n_features = len(axis), len(features)

        def kernel_init_wrap(rng, shape, dtype=jnp.float32):
            flat_shape = (
                math.prod(shape[0:n_axis]),
                math.prod(shape[-n_features:]),
            )
            flat_shape = jax.tree_map(int, flat_shape)
            kernel = self.kernel_init(rng, flat_shape, dtype)
            if isinstance(kernel, meta.AxisMetadata):
                return meta.replace_boxed(kernel, jnp.reshape(kernel.unbox(), shape))
            return jnp.reshape(kernel, shape)

        expanded_batch_shape = tuple(
            1 for ax in range(inputs.ndim) if ax not in axis
        )
        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
        kernel = self.param(
            "kernel", kernel_init_wrap, kernel_shape, self.param_dtype
        )

        contract_ind = tuple(range(0, n_axis))

        if self.use_bias:

            def bias_init_wrap(rng, shape, dtype=jnp.float32):
                flat_shape = (
                    math.prod(shape[-n_features:]),
                )
                bias = self.bias_init(rng, flat_shape, dtype)
                if isinstance(bias, meta.AxisMetadata):
                    return meta.replace_boxed(bias, jnp.reshape(bias.unbox(), shape))
                return jnp.reshape(bias, shape)

            bias = self.param(
                "bias", bias_init_wrap, features, self.param_dtype
            )
        else:
            bias = None

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        out = lax.dot_general(
            inputs,
            kernel,
            ((axis, contract_ind), ((), ())),
        )
        # dot_general output has shape [batch_dims/group_dims] + [feature_dims]
        if self.use_bias:
            # expand bias shape to broadcast bias over batch dims.
            bias = jnp.reshape(bias, expanded_batch_shape + features)
            out += bias
        return out


class MultiHeadDotProductAttention(Module):

    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.
    deterministic: Optional[bool] = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    use_bias: bool = True
    decode: bool = False

    @compact
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

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, num_attention_headss, n_features_per_head]
        query, key, value = (dense(name='query')(inputs_q),
                             dense(name='key')(inputs_kv),
                             dense(name='value')(inputs_kv))

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
        if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
            m_deterministic = merge_param('deterministic', self.deterministic,
                                          deterministic)
            if not m_deterministic:
                dropout_rng = self.make_rng('dropout')
        else:
            m_deterministic = True

        # apply attention
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
        # back to the original inputs dimensions
        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='out',  # type: ignore[call-arg]
        )(x)
        return out


class SelfAttention(MultiHeadDotProductAttention):
    """Self-attention special case of multi-head dot-product attention."""

    @compact
    def __call__(self, inputs_q: Array, mask: Optional[Array] = None,  # type: ignore
                 deterministic: Optional[bool] = None):
        return super().__call__(inputs_q, inputs_q, mask,
                                deterministic=deterministic)
