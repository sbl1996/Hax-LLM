import functools
import math
from typing import Callable, Sequence, Union, Optional, Tuple

import jax
from jax import lax, random
import jax.numpy as jnp

import flax.linen as nn
from flax.core import meta
from flax.linen import initializers
from flax.linen.attention import Dtype, Array, combine_masks, PRNGKey, Shape
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import default_embed_init, default_kernel_init

from haxllm.model.modules import _canonicalize_tuple
from haxllm.model.parallel import (
    ShardModule,
    DenseGeneral,
    ShardAxis,
    precompute_freqs_cis,
    apply_rotary_pos_emb,
)


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
            embed = self.param(
                "embed", self.inif_fn, (self.seq_len, hidden_size), self.param_dtype
            )
            embed = jnp.tile(
                embed[None], (inputs.shape[0],) + (1,) * embed.ndim
            ).astype(self.dtype)

            dense = functools.partial(
                DenseGeneral,
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
                    return meta.replace_boxed(
                        kernel, jnp.reshape(kernel.unbox(), shape)
                    )
                return jnp.reshape(kernel, shape)

            shape = (self.seq_len,) + features
            key = self.param("key", init_wrap, shape, self.param_dtype)
            value = self.param("value", init_wrap, shape, self.param_dtype)
            key = jnp.tile(key[None], (inputs.shape[0],) + (1,) * key.ndim).astype(
                self.dtype
            )
            value = jnp.tile(
                value[None], (inputs.shape[0],) + (1,) * value.ndim
            ).astype(self.dtype)
        return key, value


# Modified from flax.linen.attention.dot_product_attention_weights
def dot_product_attention_weights(
    query: Array,
    key: Array,
    gate: Optional[Array] = None,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    prefix_len: Optional[int] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
):
    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim, "q, k must have same rank."
    assert query.shape[:-3] == key.shape[:-3], "q, k batch dims must match."
    assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    attn_weights = jnp.einsum("...qhd,...khd->...hqk", query, key)

    if bias is not None:
        attn_weights = attn_weights + bias
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    if gate is None:
        attn_weights = jax.nn.softmax(attn_weights).astype(dtype)
    else:
        if prefix_len is None:
            q_len, k_len = attn_weights.shape[-2:]
            prefix_len = k_len - q_len
        attn_weights1 = jax.nn.softmax(attn_weights[..., :prefix_len]) * gate
        attn_weights2 = jax.nn.softmax(attn_weights[..., prefix_len:])
        attn_weights = jnp.concatenate([attn_weights1, attn_weights2], axis=-1).astype(
            dtype
        )

    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    return attn_weights


# Modified from flax.linen.attention.dot_product_attention
def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    prefix_gate: Optional[Array] = None,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    prefix_len: Optional[int] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
):
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # compute attention weights
    attn_weights = dot_product_attention_weights(
        query,
        key,
        prefix_gate,
        bias,
        mask,
        prefix_len,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        dtype,
    )

    # return weighted sum over values for each query position
    return jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)


# Modified from haxllm.model.parallel.SelfAttention
class SelfAttention(ShardModule):
    num_heads: int
    max_len: int
    dtype: Optional[Dtype] = None
    param_dtype: Optional[Dtype] = jnp.float32
    broadcast_dropout: bool = False
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    use_bias: bool = True
    decode: bool = False
    zero_init: bool = False
    rope: bool = False
    qkv_shard_axes: Tuple[ShardAxis, ShardAxis, ShardAxis] = ("X", "Y", None)
    out_shard_axes: Tuple[ShardAxis, ShardAxis, ShardAxis] = ("Y", None, "X")
    shard: bool = True

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Optional[Array] = None,
        prefix_key_value: Optional[Tuple[Array]] = None,
    ):
        features = x.shape[-1]
        assert (
            features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
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
            qkv_constraint(dense(name="query")(x)),
            qkv_constraint(dense(name="key")(x)),
            qkv_constraint(dense(name="value")(x)),
        )

        if prefix_key_value is not None:
            key = jnp.concatenate([prefix_key_value[0], key], axis=1)
            value = jnp.concatenate([prefix_key_value[1], value], axis=1)
            prefix_len = prefix_key_value[0].shape[1]
        else:
            prefix_len = None

        if self.rope:
            cos, sin = precompute_freqs_cis(
                dim=head_dim, end=self.max_len, dtype=self.dtype
            )

        if not self.decode:
            if self.rope:
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
        else:
            is_initialized = self.has_variable("cache", "cached_key")
            init_fn = jnp.zeros
            if self.shard:
                init_fn = nn.with_partitioning(init_fn, self.qkv_shard_axes)
            cached_key = self.variable(
                "cache", "cached_key", init_fn, key.shape, key.dtype
            )
            cached_value = self.variable(
                "cache", "cached_value", init_fn, value.shape, value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )
            if not is_initialized:
                if self.rope:
                    query, key = apply_rotary_pos_emb(query, key, cos, sin)
            else:
                (
                    *batch_dims,
                    max_length,
                    num_heads,
                    depth_per_head,
                ) = cached_key.value.shape
                # shape check of cached keys against query input
                expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        "expected query shape %s instead got %s."
                        % (expected_shape, query.shape)
                    )
                # update key, value caches with our new 1d spatial slices
                cur_index = cache_index.value
                if self.rope:
                    pos_index = jnp.array([cur_index], dtype=jnp.int32)
                    cos, sin = cos[pos_index], sin[pos_index]
                    query, key = apply_rotary_pos_emb(query, key, cos, sin)
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(max_length) <= cur_index,
                        tuple(batch_dims) + (1, 1, max_length),
                    ),
                )

        dropout_rng = None
        if self.dropout_rate > 0 and not self.deterministic:
            dropout_rng = self.make_rng("dropout")
            deterministic = False
        else:
            deterministic = True

        if self.zero_init:
            prefix_gate = self.param(
                "prefix_gate", initializers.zeros, (self.num_heads,), self.param_dtype
            )
            prefix_gate = prefix_gate[None, :, None, None]
        else:
            prefix_gate = None
        x = dot_product_attention(
            query,
            key,
            value,
            prefix_gate=prefix_gate,
            mask=mask,
            prefix_len=prefix_len,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
        )

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
            name="out",
        )(
            x
        )  # type: ignore
        # out = self.with_sharding_constraint(
        #     out, ("X", None, "Y"))
        return out
