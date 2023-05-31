import math
import functools
from typing import Callable, Optional, Tuple, Union, Sequence, Iterable, Any

import jax
import jax.numpy as jnp
from jax import lax

from flax.core import meta
import flax.linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen import initializers
from flax.linen.attention import Dtype, Array, PRNGKey, Shape
from flax.linen.linear import default_embed_init

from haxllm.model.efficient_attention import dot_product_attention as dot_product_attention_m
from haxllm.gconfig import get_remat_policy
from haxllm.config_utils import RematScanConfigMixin


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


class PrefixEmbed(nn.Module):
    seq_len: int
    features: Union[int, Sequence[int]]
    projection: bool = False
    prefix_features: int = 512
    inif_fn: Callable = default_embed_init
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        features = _canonicalize_tuple(self.features)
        n_features = len(features)
        if self.projection:
            hidden_size = math.prod(features)
            embed = self.param("embed", self.inif_fn, (self.seq_len, hidden_size), self.param_dtype)
            embed = jnp.tile(embed[None], (inputs.shape[0],) + (1,) * embed.ndim).astype(self.dtype)

            dense = functools.partial(
                nn.DenseGeneral,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
            x = dense(features=self.prefix_features, name="trans1")(embed)
            x = nn.tanh(x)
            key = dense(features=features, name="trans2_key")(x)
            value = dense(features=features, name="trans2_value")(x)
        else:
            def init_wrap(rng, shape, dtype=jnp.float32):
                flat_shape = (
                    math.prod(shape[0:1]),
                    math.prod(shape[-n_features:]),
                )
                flat_shape = jax.tree_map(int, flat_shape)
                kernel = self.inif_fn(rng, flat_shape, dtype)
                if isinstance(kernel, meta.AxisMetadata):
                    return meta.replace_boxed(kernel, jnp.reshape(kernel.unbox(), shape))
                return jnp.reshape(kernel, shape)
            shape = (self.seq_len,) + features
            key = self.param("key", init_wrap, shape, self.param_dtype)
            value = self.param("value", init_wrap, shape, self.param_dtype)
            key = jnp.tile(key[None], (inputs.shape[0],) + (1,) * key.ndim).astype(self.dtype)
            value = jnp.tile(value[None], (inputs.shape[0],) + (1,) * value.ndim).astype(self.dtype)
        return key, value


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


class MlpBlock(nn.Module):
    intermediate_size: Optional[int] = None
    activation: str = 'gelu'
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs):
        assert self.activation in ['gelu', 'gelu_new']
        intermediate_size = self.intermediate_size or 4 * inputs.shape[-1]

        dense = functools.partial(
            nn.Dense,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        actual_out_dim = inputs.shape[-1]
        x = dense(intermediate_size, name="fc_1")(inputs)
        if self.activation == 'gelu':
            x = nn.gelu(x, approximate=False)
        elif self.activation == 'gelu_new':
            x = nn.gelu(x, approximate=True)
        x = dense(actual_out_dim, name="fc_2")(x)
        return x


def make_block_stack(block_fn, n_layers, config: RematScanConfigMixin):
    from haxllm.model.parallel import remat_scan, remat
    remat_policy = get_remat_policy()

    def stack_fn(x, train):
        block_fn_ = block_fn
        if config.remat_scan:
            remat_scan_lengths = config.remat_scan_lengths()
            if len(remat_scan_lengths) == 3:
                n_loop = remat_scan_lengths[0]
                lengths = remat_scan_lengths[1:]
            else:
                n_loop = None
                lengths = remat_scan_lengths
            TransformerBlockStack = remat_scan(
                block_fn_, lengths=lengths, policy=remat_policy,
                variable_axes={True: 0}, split_rngs={True: True}, metadata_params={nn.PARTITION_NAME: None})
            if n_loop is not None:
                for i in range(n_loop):
                    x = TransformerBlockStack(config=config, name=f'hs_{i}')(x)
            else:
                x = TransformerBlockStack(config=config, name='hs')(x)
        else:
            if config.scan:
                if config.remat and train:
                    block_fn_ = remat(block_fn_, prevent_cse=False, policy=remat_policy)
                TransformerBlockStack = nn.scan(
                    block_fn_, length=config.scan_lengths()[0], variable_axes={True: 0},
                    split_rngs={True: True}, metadata_params={nn.PARTITION_NAME: None})
                x = TransformerBlockStack(config=config, scan=True, name='hs')(x)[0]
            else:
                if config.remat and train:
                    block_fn_ = remat(block_fn_, policy=remat_policy)
                for i in range(n_layers):
                    x = block_fn_(config=config, name=f'h_{i}')(x)
        return x
    return stack_fn