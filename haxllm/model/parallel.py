from typing import Mapping, Optional, Tuple, Callable, Sequence, Union
import functools
import dataclasses

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.core import lift
from flax.linen import partitioning as nn_partitioning, initializers
from flax.linen.attention import (
    Dtype,
    Array,
    combine_masks,
    PRNGKey,
    Shape,
    dot_product_attention,
)

from haxllm.model.modules import DenseGeneral
from haxllm.gconfig import get as get_gconfig
from haxllm.model.efficient_attention import dot_product_attention as dot_product_attention_m


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


class DenseGeneral(ShardMixIn, DenseGeneral):
    pass


class Embed(ShardMixIn, nn.Embed):
    pass


ShardAxis = Optional[str]


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32
):
    # returns:
    #   cos, sin: (end, dim)
    freqs = 1.0 / (
        theta ** (np.arange(0, dim, 2, dtype=np.float32)[: (dim // 2)] / dim)
    )
    t = np.arange(end, dtype=np.float32)  # type: ignore
    freqs = np.outer(t, freqs).astype(dtype)  # type: ignore
    freqs = np.concatenate((freqs, freqs), axis=-1)
    cos, sin = np.cos(freqs), np.sin(freqs)
    return jnp.array(cos, dtype=dtype), jnp.array(sin, dtype=dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # inputs:
    #   x: (batch_size, seq_len, num_heads, head_dim)
    #   cos, sin: (seq_len, head_dim)
    # returns:
    #   q, k: (batch_size, seq_len, num_heads, head_dim)
    q_len = q.shape[1]
    kv_len = k.shape[1]
    prefix_len = kv_len - q_len

    cos_k = cos[None, :kv_len, None, :]
    sin_k = sin[None, :kv_len, None, :]
    k = (k * cos_k) + (rotate_half(k) * sin_k)

    cos_q = cos_k[:, prefix_len:]
    sin_q = sin_k[:, prefix_len:]
    q = (q * cos_q) + (rotate_half(q) * sin_q)
    return q, k


def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # inputs:
    #   x: (batch_size, seq_len, num_heads, head_dim)
    #   cos, sin: (seq_len, head_dim)
    #   position_id: (batch_size, seq_len)
    # returns:
    #   x: (batch_size, seq_len, num_heads, head_dim)
    cos = jnp.take(cos, position_id, axis=0)[:, :, None, :]
    sin = jnp.take(sin, position_id, axis=0)[:, :, None, :]

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def apply_glm_rotary_pos_emb(q, k, cos, sin, position_ids):
    q1, q2 = jnp.array_split(q, 2, axis=-1)
    k1, k2 = jnp.array_split(k, 2, axis=-1)
    block_position_ids = position_ids[:, 1, :]
    position_ids = position_ids[:, 0, :]
    q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
    q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
    q = jnp.concatenate([q1, q2], axis=-1)
    k = jnp.concatenate([k1, k2], axis=-1)
    return q, k


ModuleClass = Callable[..., nn.Module]

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
    rope: bool = False
    memory_efficient: bool = False
    memory_efficient_mask_mode: str = "causal"
    qkv_shard_axes: Tuple[ShardAxis, ShardAxis, ShardAxis] = ("X", "Y", None)
    out_shard_axes: Tuple[ShardAxis, ShardAxis, ShardAxis] = ("Y", None, "X")
    shard: bool = True
    dense_cls: Union[ModuleClass, Sequence[ModuleClass]] = DenseGeneral

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Optional[Array] = None,
        position_ids: Optional[Array] = None,
    ):
        features = x.shape[-1]
        assert (
            features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = features // self.num_heads

        if position_ids is not None:
            assert position_ids.ndim == 3, "Only ChatGLM style 2D position_ids can be used."

        if not isinstance(self.dense_cls, Sequence):
            dense_cls = [self.dense_cls for _ in range(4)]
        else:
            assert len(self.dense_cls) == 4, "dense_cls must be a sequence of length 4 for query, key, value, and out."
            dense_cls = self.dense_cls

        qkv_dense = [
            functools.partial(
                cls,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                features=(self.num_heads, head_dim),
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.use_bias,
                shard_axes={"kernel": self.qkv_shard_axes},
                shard=self.shard,
                axis=-1,
            ) for cls in dense_cls[:3]
        ]

        qkv_constraint = lambda x: x
        # qkv_constraint = functools.partial(
        #     self.with_sharding_constraint, axes=("X", None, "Y", None))

        query, key, value = (
            qkv_constraint(qkv_dense[0](name="query")(x)),
            qkv_constraint(qkv_dense[1](name="key")(x)),
            qkv_constraint(qkv_dense[2](name="value")(x)),
        )

        # ChatGLM style 2d position encoding
        position_encoding_2d = position_ids is not None

        if self.rope:
            pe_dim = head_dim // 2 if position_encoding_2d else head_dim
            cos, sin = precompute_freqs_cis(
                dim=pe_dim, end=self.max_len, dtype=self.dtype)

        if not self.decode:
            if self.rope:
                if position_encoding_2d:
                    # 2D position_ids is (batch_size, 2, seq_len)
                    query, key = apply_glm_rotary_pos_emb(query, key, cos, sin, position_ids)
                else:
                    query, key = apply_rotary_pos_emb(query, key, cos, sin)
        else:
            is_initialized = self.has_variable("cache", "cached_key")
            init_fn = jnp.zeros
            # if self.shard:
            #     init_fn = nn.with_partitioning(init_fn, self.qkv_shard_axes)
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
                    if position_encoding_2d:
                        # 2D position_ids is (batch_size, 2, seq_len)
                        query, key = apply_glm_rotary_pos_emb(query, key, cos, sin, position_ids)
                    else:
                        query, key = apply_rotary_pos_emb(query, key, cos, sin)
            else:
                *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
                # shape check of cached keys against query input
                # expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
                # if expected_shape != query.shape:
                #     raise ValueError(
                #         "Autoregressive cache shape error, "
                #         "expected query shape %s instead got %s."
                #         % (expected_shape, query.shape)
                #     )
                # update key, value caches with our new 1d spatial slices
                num_queries = query.shape[-3]
                cur_index = cache_index.value
                if self.rope:
                    if position_encoding_2d:
                        query, key = apply_glm_rotary_pos_emb(query, key, cos, sin, position_ids)
                    else:
                        position_ids = jnp.arange(num_queries) + cur_index
                        position_ids = jnp.broadcast_to(position_ids, tuple(batch_dims) + (num_queries,))
                        query, key = apply_rotary_pos_emb_index(query, key, cos, sin, position_ids)
                indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                if num_queries > 1:
                    idx = jnp.arange(max_length)
                    mask = idx <= jnp.arange(num_queries)[:, None]
                    if position_encoding_2d:
                        # First stage of decoding of ChatGLM
                        context_length_1 = position_ids[0, 0, -1]
                        mask = mask | (idx <= context_length_1)
                        offset = context_length_1 + 2
                    else:
                        offset = num_queries
                    cache_index.value = cache_index.value + offset
                    mask = jnp.broadcast_to(
                        mask, tuple(batch_dims) + (1, num_queries, max_length),
                    )
                else:
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

        if self.memory_efficient:
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
            x = dot_product_attention(
                query,
                key,
                value,
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
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            shard_axes={"kernel": self.out_shard_axes},
            shard=self.shard,
            name="out",
        )(x)

        # out = self.with_sharding_constraint(
        #     out, ("X", None, "Y"))
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
    dense_cls: Union[ModuleClass, Sequence[ModuleClass]] = DenseGeneral

    @nn.compact
    def __call__(self, inputs):
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
            name="gate",
        )(inputs)
        g = nn.silu(g)
        # g = self.with_sharding_constraint(
        #     g, ("X", None, "Y"))
        x = g * dense[1](
            features=self.intermediate_size,
            shard_axes={"kernel": self.shard_axes1},
            name="up",
        )(inputs)
        # x = self.with_sharding_constraint(
        #     x, ("X", None, "Y"))
        x = dense[2](
            features=actual_out_dim,
            shard_axes={"kernel": self.shard_axes2},
            name="down",
        )(x)
        # x = self.with_sharding_constraint(
        #     x, ("X", None, "Y"))
        return x
