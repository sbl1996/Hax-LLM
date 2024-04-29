import functools
from typing import Callable, Sequence, Union, Optional, Tuple

import jax.numpy as jnp

import flax.linen as nn
from flax.linen import initializers
from flax.linen.attention import Dtype, Array, PRNGKey, Shape
from flax.linen.linear import default_kernel_init

from haxllm.model.parallel import (
    ShardModule,
    DenseGeneral,
    ShardAxis,
    replicate_for_multi_query,
    ModuleClass,
)
from haxllm.model.ptuning.modules import PrefixEmbed, dot_product_attention
from haxllm.model.attention import decode_for_padding_left, decode_for_padding_right, get_position_ids_for_padding_left, make_apply_rope, init_decode_cache


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
    padding_left: bool = False
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
        padding_mask: Optional[Array] = None,
        prefix_key_value: Optional[Tuple[Array, Array]] = None,
    ):
        r"""
        Parameters
        ----------
        x: Array, shape [batch, q_len, features]
            Input features.
        mask: Optional[Array], shape [batch, 1, q_len, kv_len]
            Mask to apply to attention scores.
        padding_mask: Optional[Array], shape [batch, q_len]
            Mask to indicate which query elements are padding.
            If both mask and padding_mask are provided, you must combine them by yourself.
            We only use padding_mask to infer position_ids.
        prefix_key_value: Optional[Tuple[Array, Array]], shape [batch, prefix_len, features]
            Prefix key and value vectors.
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
        assert prefix_key_value is not None, "prefix_key_value must be provided."
        for kv in prefix_key_value:
            assert kv.shape[1] == self.prefix_len, "prefix_key_value must have the same prefix length as the module."
        if self.padding_left:
            if x.shape[1] > 1:
                assert padding_mask is not None, "padding_mask must be provided for the first stage of decoding with padding_left."

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
            kv_dense_shard_axes = None
        else:
            kv_dense_shard_axes = {"kernel": kv_shard_axes}


        kv_dense = [
            functools.partial(
                cls,
                features=(kv_num_heads, head_dim),
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

        k_dense = kv_dense[0](name="key")
        v_dense = kv_dense[1](name="value")

        key = k_dense(x)
        value = v_dense(x)

        if multi_query:
            key = replicate_for_multi_query(key, self.num_heads)
            value = replicate_for_multi_query(value, self.num_heads)

        if self.rope:
            add_pos = make_apply_rope(head_dim, self.max_len, self.dtype, self.rope_theta, variant=2 if self.rope == 2 else 1)
        else:
            add_pos = lambda q, k, p=None: (q, k)

        if not self.decode:
            if self.padding_left:
                position_ids = get_position_ids_for_padding_left(padding_mask)
            else:
                position_ids = None
            query, key = add_pos(query, key, position_ids)
        else:
            assert mask is None, "Mask is not needed for decoding, we infer it from cache."
            is_initialized, cached_key, cached_value, cache_index = init_decode_cache(self, key, value, kv_shard_axes)
            if self.padding_left:
                cache_position = self.variable(
                    "cache", "cache_position", lambda: jnp.zeros(key.shape[0], dtype=jnp.int32)
                )

            if is_initialized:
                if self.padding_left:
                    query, key, value, mask = decode_for_padding_left(
                        add_pos, query, key, value, cache_index, cached_key, cached_value, cache_position, padding_mask)
                else:
                    query, key, value, mask = decode_for_padding_right(
                        add_pos, query, key, value, cache_index, cached_key, cached_value)


        adapter_k, adapter_v = prefix_key_value
        adapter_k = k_dense(adapter_k)
        adapter_v = v_dense(adapter_v)
        if multi_query:
            adapter_k = replicate_for_multi_query(adapter_k, self.num_heads)
            adapter_v = replicate_for_multi_query(adapter_v, self.num_heads)
        key = jnp.concatenate([adapter_k, key], axis=1)
        value = jnp.concatenate([adapter_v, value], axis=1)
        if not self.decode:
            assert mask is not None, "causal mask is required for training."
        if mask is not None:
            prefix_mask = jnp.ones(mask.shape[:-1] + (self.prefix_len,), dtype=mask.dtype)
            mask = jnp.concatenate([prefix_mask, mask], axis=-1)


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
            assert not self.padding_left, "Memory efficient attention does not support padding_left."
        
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