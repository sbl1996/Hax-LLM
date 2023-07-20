import functools
from typing import Callable, Sequence, Union, Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp

import flax.linen as nn
from flax.linen import initializers
from flax.linen.attention import Dtype, Array, PRNGKey, Shape
from flax.linen.linear import default_kernel_init

from haxllm.model.parallel import (
    ShardModule,
    DenseGeneral,
    ShardAxis,
    precompute_freqs_cis,
    apply_rotary_pos_emb_index,
    precompute_freqs_cis2,
    apply_rotary_pos_emb_index2,
    replicate_for_multi_query,
    ModuleClass,
)
from haxllm.model.ptuning.modules import PrefixEmbed, dot_product_attention


# Shape with longer sequence length
def _longer_shape(shape, extra_length, axis):
    return shape[:axis] + (shape[axis] + extra_length,) + shape[axis + 1:]


# Modified from haxllm.model.parallel.SelfAttention
class SelfAttention(ShardModule):
    num_heads: int
    max_len: int
    prefix_len: int
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
    refresh_cache: bool = False
    dense_cls: Union[ModuleClass, Sequence[ModuleClass]] = DenseGeneral

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Optional[Array] = None,
        prefix_key_value: Optional[Tuple[Array, Array]] = None,
    ):
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

        k_dense = kv_dense[0](name="key")
        v_dense = kv_dense[1](name="value")

        key = k_dense(x)
        value = v_dense(x)

        if multi_query:
            key = replicate_for_multi_query(key, self.num_heads)
            value = replicate_for_multi_query(value, self.num_heads)

        if self.rope:
            if multi_query:
                cos, sin = precompute_freqs_cis2(
                    dim=head_dim // 2, end=self.max_len, dtype=self.dtype)
                add_pos = lambda q, k, p=None: apply_rotary_pos_emb_index2(q, k, cos, sin, p)
            else:
                cos, sin = precompute_freqs_cis(
                    dim=head_dim, end=self.max_len, dtype=self.dtype)
                add_pos = lambda q, k, p=None: apply_rotary_pos_emb_index(q, k, cos, sin, p)
        else:
            add_pos = lambda q, k, p=None: (q, k)

        if not self.decode:
            # Training
            assert prefix_key_value is not None
            query, key = add_pos(query, key)

            adapter_k, adapter_v = prefix_key_value
            adapter_k = k_dense(adapter_k)
            adapter_v = v_dense(adapter_v)
            if multi_query:
                adapter_k = replicate_for_multi_query(adapter_k, self.num_heads)
                adapter_v = replicate_for_multi_query(adapter_v, self.num_heads)

            key = jnp.concatenate([adapter_k, key], axis=1)
            value = jnp.concatenate([adapter_v, value], axis=1)
            prefix_len = adapter_k.shape[1]
            assert prefix_len == self.prefix_len

            assert mask is not None, "causal mask is required for training."
            prefix_mask = jnp.ones(mask.shape[:-1] + (prefix_len,), dtype=mask.dtype)
            mask = jnp.concatenate([prefix_mask, mask], axis=-1)
        else:
            is_initialized = self.has_variable("cache", "cached_key")
            init_fn = jnp.zeros
            if self.shard_cache:
                init_fn = nn.with_partitioning(init_fn, kv_shard_axes)
            prefix_len = self.prefix_len
            
            cached_key = self.variable(
                "cache", "cached_key", init_fn, _longer_shape(key.shape, prefix_len, -3), key.dtype
            )
            cached_value = self.variable(
                "cache", "cached_value", init_fn, _longer_shape(value.shape, prefix_len, -3), value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )
            if not is_initialized:
                assert prefix_key_value is not None
                query, key = add_pos(query, key)

                adapter_k, adapter_v = prefix_key_value
                adapter_k = k_dense(adapter_k)
                adapter_v = v_dense(adapter_v)
                if multi_query:
                    adapter_k = replicate_for_multi_query(adapter_k, self.num_heads)
                    adapter_v = replicate_for_multi_query(adapter_v, self.num_heads)

                key = jnp.concatenate([adapter_k, key], axis=1)
                value = jnp.concatenate([adapter_v, value], axis=1)
                prefix_len = adapter_k.shape[1]
                assert prefix_len == self.prefix_len
            elif self.refresh_cache:
                assert prefix_key_value is not None
                adapter_k, adapter_v = prefix_key_value
                adapter_k = k_dense(adapter_k)
                adapter_v = v_dense(adapter_v)
                if multi_query:
                    adapter_k = replicate_for_multi_query(adapter_k, self.num_heads)
                    adapter_v = replicate_for_multi_query(adapter_v, self.num_heads)

                *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
                indices = (0,) * len(batch_dims) + (0, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, adapter_k, indices)
                value = lax.dynamic_update_slice(cached_value.value, adapter_v, indices)
                cached_key.value = key
                cached_value.value = value
            else:
                *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
                num_queries = query.shape[-3]
                cur_index = cache_index.value
                offset = cur_index + prefix_len

                position_ids = jnp.arange(num_queries) + cur_index
                position_ids = jnp.broadcast_to(position_ids, tuple(batch_dims) + (num_queries,))
                query, key = add_pos(query, key, position_ids)
                
                indices = (0,) * len(batch_dims) + (offset, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value

                cache_index.value = cache_index.value + num_queries

                idx = jnp.arange(num_queries) + offset
                mask = jnp.arange(max_length)[None, :] <= idx[:, None]
                mask = jnp.broadcast_to(
                    mask, tuple(batch_dims) + (1, num_queries, max_length),
                )

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
                prefix_len=self.prefix_len,
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
