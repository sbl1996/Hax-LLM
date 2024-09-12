import functools
import math
from typing import Callable, Sequence, Union, Optional, Tuple

import jax
from jax import lax, random
import jax.numpy as jnp

import flax.linen as nn
from flax.core import meta
from flax.linen import initializers
from flax.linen.attention import Dtype, Array, PRNGKey, Shape
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import default_embed_init, default_kernel_init

from haxllm.model.modules import _canonicalize_tuple
from haxllm.model.parallel import (
    ShardModule,
    DenseGeneral,
    ShardAxis,
    ModuleClass,
    replicate_for_multi_query
)
from haxllm.model.attention import decode_for_padding_right, make_apply_rope


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
                flat_shape = jax.tree.map(int, flat_shape)
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
    prefix_len: int,
    gate: Optional[Array] = None,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
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
        gate = jnp.tanh(gate)
        attn_weights1 = jax.nn.softmax(attn_weights[..., :prefix_len]) * gate
        attn_weights2 = jax.nn.softmax(attn_weights[..., prefix_len:])
        attn_weights = jnp.concatenate([attn_weights1, attn_weights2], axis=-1).astype(dtype)

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
    prefix_len: int,
    prefix_gate: Optional[Array] = None,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
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
        prefix_len,
        prefix_gate,
        bias,
        mask,
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
    multi_query_groups: Optional[int] = None
    dtype: Optional[Dtype] = None
    param_dtype: Optional[Dtype] = jnp.float32
    broadcast_dropout: bool = False
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    qkv_bias: bool = True
    out_bias: bool = True
    decode: bool = False
    rope: bool = False
    zero_init: bool = True
    memory_efficient: bool = False
    memory_efficient_mask_mode: str = "causal"
    query_shard_axes: Tuple[ShardAxis, ShardAxis, ShardAxis] = ("X", "Y", None)
    kv_shard_axes: Optional[Tuple[ShardAxis, ShardAxis, ShardAxis]] = None
    out_shard_axes: Tuple[ShardAxis, ShardAxis, ShardAxis] = ("Y", None, "X")
    shard: bool = True
    shard_cache: bool = False
    dense_cls: Union[ModuleClass, Sequence[ModuleClass]] = DenseGeneral

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Optional[Array] = None,
        prefix_key_value: Optional[Tuple[Array]] = None,
    ):
        assert prefix_key_value is not None
        multi_query = self.multi_query_groups is not None
        kv_shard_axes = self.kv_shard_axes or self.query_shard_axes
        features = x.shape[-1]
        assert (
            features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = features // self.num_heads

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
            axis=-1,
            name="query",
        )(x)

        kv_num_heads = self.num_heads
        if multi_query:
            kv_num_heads = self.multi_query_groups

        kv_dense = [
            functools.partial(
                cls,
                features=(kv_num_heads, head_dim),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.qkv_bias,
                shard_axes={"kernel": kv_shard_axes},
                shard=self.shard,
                axis=-1,
            ) for cls in dense_cls[1:3]
        ]

        key = kv_dense[0](name="key")(x)
        value = kv_dense[1](name="value")(x)

        if multi_query:
            key = replicate_for_multi_query(key, self.num_heads)
            value = replicate_for_multi_query(value, self.num_heads)

        if self.rope:
            add_pos = make_apply_rope(head_dim, self.max_len, self.dtype, self.rope_theta, variant=2 if self.rope == 2 else 1)
        else:
            add_pos = lambda q, k, p=None: (q, k)

        if not self.decode:
            query, key = add_pos(query, key)
        else:
            assert mask is None, "Mask is not needed for decoding, we infer it from cache."
            is_initialized = self.has_variable("cache", "cached_key")
            init_fn = jnp.zeros
            if self.shard and self.shard_cache:
                init_fn = nn.with_partitioning(init_fn, kv_shard_axes)
            cached_key = self.variable(
                "cache", "cached_key", init_fn, key.shape, key.dtype
            )
            cached_value = self.variable(
                "cache", "cached_value", init_fn, value.shape, value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )

            if is_initialized:
                query, key, value, mask = decode_for_padding_right(
                    add_pos, query, key, value, cache_index, cached_key, cached_value)

        dropout_rng = None
        if self.dropout_rate > 0 and not self.deterministic:
            dropout_rng = self.make_rng("dropout")
            deterministic = False
        else:
            deterministic = True

        if self.memory_efficient:
            raise NotImplementedError
            assert not self.decode, "Memory efficient attention does not support decoding."
            assert deterministic, "Memory efficient attention does not support dropout."
        
            mask_mode = self.memory_efficient_mask_mode
            context_lengths = None
            pad_positions = None
            if mask_mode == "causal":
                if mask is not None:
                    print("WARNING: mask is not needed for memory efficient attention using mask_mode='causal'.")
                mask = None
            # TODO: implement padding mask
            elif mask_mode == 'padding':
                raise NotImplementedError
            #     if mask is not None:
            #         print("WARNING: padding mask is needed for memory efficient attention using mask_mode='padding'.")
            #     mask = mask[:, None, None, :]
            elif mask_mode == 'bidirectional':
                if mask is not None:
                    print("WARNING: mask is not needed for memory efficient attention using mask_mode='bidirectional', we infer it from position_ids.")
                    mask = None
                context_lengths = jnp.argmax(position_ids[:, 0, :], axis=1) + 1
            x = dot_product_attention_m(
                query,
                key,
                value,
                pad_positions,
                context_lengths,
                mask_mode,
                dtype=self.dtype,
            )
        else:
            key = jnp.concatenate([prefix_key_value[0], key], axis=1)
            value = jnp.concatenate([prefix_key_value[1], value], axis=1)
            prefix_len = prefix_key_value[0].shape[1]
            if mask is not None:
                prefix_mask = jnp.ones(mask.shape[:-1] + (prefix_len,), dtype=mask.dtype)
                mask = jnp.concatenate([prefix_mask, mask], axis=-1)

            if self.zero_init:
                prefix_gate = self.param(
                    "prefix_gate", initializers.zeros, (self.num_heads,), jnp.float32,
                )
                prefix_gate = prefix_gate[None, :, None, None]
            else:
                prefix_gate = None

            x = dot_product_attention(
                query,
                key,
                value,
                prefix_len=prefix_len,
                prefix_gate=prefix_gate,
                mask=mask,
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
            name="out",
        )(x)
        return out
