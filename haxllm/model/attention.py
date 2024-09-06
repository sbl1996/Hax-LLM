from typing import Optional, Callable
import math
import numpy as np

import jax
from jax import lax
import jax.numpy as jnp

import flax.linen as nn
from flax.linen.attention import (
    Dtype,
    Array,
    PRNGKey,
    PrecisionLike,
)
from flax.linen.dtypes import promote_dtype
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

from haxllm.model.mixin import RoPEScalingConfig


def compute_inv_freq(dim: int, base = 10000.0, dtype=np.float32):
    return 1.0 / (base ** (np.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))


def compute_llama3_inv_freq(
    dim: int, base = 10000.0, factor: float = 8.0,
    low_freq_factor: float = 1.0, high_freq_factor: float = 4.0,
    max_position_embeddings: int = 8192):
    inv_freq = compute_inv_freq(dim, base)
    old_context_len = max_position_embeddings

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama


def precompute_freqs_cis(inv_freq, end, dtype = jnp.float32):
    # returns:
    #   cos, sin: (end, dim)
    freqs = inv_freq
    t = np.arange(end, dtype=freqs.dtype)  # type: ignore
    freqs = np.outer(t, freqs)  # type: ignore
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
    cos_q = cos_k[:, prefix_len:]
    sin_q = sin_k[:, prefix_len:]

    q = (q * cos_q) + (rotate_half(q) * sin_q)
    k = (k * cos_k) + (rotate_half(k) * sin_k)
    return q, k


def apply_rotary_pos_emb_index(q, k, cos, sin, position_id=None):
    # inputs:
    #   x: (batch_size, seq_len, num_heads, head_dim)
    #   cos, sin: (seq_len, head_dim)
    #   position_id: (batch_size, seq_len)
    # returns:
    #   x: (batch_size, seq_len, num_heads, head_dim)
    if position_id is None:
        q_pos = jnp.arange(q.shape[1])[None, :]
        k_pos = jnp.arange(k.shape[1])[None, :]
    else:
        q_pos = position_id
        k_pos = position_id

    cos_q = jnp.take(cos, q_pos, axis=0)[:, :, None, :]
    sin_q = jnp.take(sin, q_pos, axis=0)[:, :, None, :]
    q = (q * cos_q) + (rotate_half(q) * sin_q)

    cos_k = jnp.take(cos, k_pos, axis=0)[:, :, None, :]
    sin_k = jnp.take(sin, k_pos, axis=0)[:, :, None, :]
    k = (k * cos_k) + (rotate_half(k) * sin_k)
    return q, k


# from chatglm, different from original rope
def precompute_freqs_cis_glm(inv_freq, end: int, dtype = jnp.float32):
    # returns:
    #   cos, sin: (end, dim)
    freqs = inv_freq
    t = np.arange(end, dtype=freqs.dtype)  # type: ignore
    freqs = np.outer(t, freqs)  # type: ignore
    cos, sin = np.cos(freqs), np.sin(freqs)
    return jnp.array(cos, dtype=dtype), jnp.array(sin, dtype=dtype)


def apply_cos_sin(x, cos, sin):
    dim = x.shape[-1]
    x1 = x[..., :dim // 2]
    x2 = x[..., dim // 2:]
    x1 = x1.reshape(x1.shape[:-1] + (-1, 2))
    x1 = jnp.stack((x1[..., 0] * cos - x1[..., 1] * sin, x1[..., 1] * cos + x1[..., 0] * sin), axis=-1)
    x1 = x1.reshape(x2.shape)
    x = jnp.concatenate((x1, x2), axis=-1)
    return x


def apply_rotary_pos_emb_index2(q, k, cos, sin, position_id=None):
    # inputs:
    #   x: (batch_size, seq_len, num_heads, head_dim)
    #   cos, sin: (seq_len, head_dim)
    #   position_id: (batch_size, seq_len)
    # returns:
    #   x: (batch_size, seq_len, num_heads, head_dim)
    if position_id is None:
        q_pos = jnp.arange(q.shape[1])[None, :]
        k_pos = jnp.arange(k.shape[1])[None, :]
    else:
        q_pos = position_id
        k_pos = position_id
    
    cos_q = jnp.take(cos, q_pos, axis=0)[:, :, None, :]
    sin_q = jnp.take(sin, q_pos, axis=0)[:, :, None, :]
    q = apply_cos_sin(q, cos_q, sin_q)

    cos_k = jnp.take(cos, k_pos, axis=0)[:, :, None, :]
    sin_k = jnp.take(sin, k_pos, axis=0)[:, :, None, :]
    k = apply_cos_sin(k, cos_k, sin_k)
    return q, k


def make_apply_rope(head_dim, max_len, dtype, base=10000.0, scaling: Optional[RoPEScalingConfig] = None):
    rope_type = "default" if scaling is None else scaling.rope_type
    if rope_type == "chatglm":
        inv_freq = compute_inv_freq(head_dim // 2, base)
        cos, sin = precompute_freqs_cis_glm(inv_freq, max_len, dtype=dtype)
        add_pos = lambda q, k, p=None: apply_rotary_pos_emb_index2(q, k, cos, sin, p)
        return add_pos
    scaling_factor = None
    if rope_type == "default":
        inv_freq = compute_inv_freq(head_dim, base)
    elif rope_type == "llama3":
        inv_freq = compute_llama3_inv_freq(
            head_dim, base, scaling.factor, scaling.low_freq_factor, scaling.high_freq_factor,
            scaling.max_position_embeddings)
    elif rope_type == "dynamic":
        orig_max_len = scaling.max_position_embeddings
        if max_len > orig_max_len:
            alpha = scaling.factor * max_len / orig_max_len - (scaling.factor - 1)
            base = base * alpha ** (head_dim / (head_dim - 2))
        inv_freq = compute_inv_freq(head_dim, base)
    elif rope_type == "longrope":
        orig_max_len = scaling.max_position_embeddings
        inv_freq = compute_inv_freq(head_dim, base)
        if max_len > orig_max_len:
            factors = np.array(scaling.long_factor, dtype=inv_freq.dtype)
        else:
            factors = np.array(scaling.short_factor, dtype=inv_freq.dtype)
        inv_freq = inv_freq * factors
        scale = max_len / orig_max_len
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(1 + math.log(scale) / math.log(orig_max_len))
    else:
        raise ValueError(f"Unknown rope type: {rope_type}")
    cos, sin = precompute_freqs_cis(inv_freq, max_len, dtype=dtype)
    if scaling_factor is not None:
        scaling_factor = np.array(scaling_factor, dtype=dtype)
        cos = cos * scaling_factor
        sin = sin * scaling_factor
    add_pos = lambda q, k, p=None: apply_rotary_pos_emb_index(q, k, cos, sin, p)
    return add_pos


def get_position_ids_for_padding_left(padding_mask, return_pad_position=False):
    r"""
    Get position ids from padding mask for padding left.

    Parameters
    ----------
    padding_mask: jnp.ndarray, shape (batch_size, seq_len)
        Mask to indicate which tokens are padding tokens.
    return_pad_position: bool, default False
        Whether to return the end position of padding tokens.

    Returns
    -------
    position_ids: jnp.ndarray, shape (batch_size, seq_len)
        Position ids.
    pad_position: jnp.ndarray, shape (batch_size,)
        End position of padding tokens.
    
    Examples
    --------
    >>> padding_mask = jnp.array([[1,0,0], [1,1,0]], dtype=jnp.bool_)
    >>> position_ids, pad_position = get_position_ids_for_padding_left(padding_mask, return_pad_position=True)
    >>> position_ids
    Array([[0, 0, 1],
           [0, 0, 0]], dtype=int32)
    >>> pad_position
    Array([1, 2], dtype=int32)
    """
    seq_len = padding_mask.shape[1]
    pad_position = jnp.argmin(padding_mask, axis=-1)
    position_ids = jnp.arange(seq_len)[None, :] - pad_position[:, None]
    position_ids = jnp.clip(position_ids, 0)
    if return_pad_position:
        return position_ids, pad_position
    return position_ids


def decode_for_padding(
    add_pos, query, key, value, cache_index, cached_key, cached_value,
    padding_left, cache_position=None, padding_mask=None, window_size=None):
    num_queries = query.shape[-3]
    cur_index = cache_index.value

    assert key.ndim == 4 and value.ndim == 4, "Only 1D batched input is supported for decoding."
    batch_size, max_length, num_heads, depth_per_head = cached_key.value.shape

    if padding_left:
        if num_queries > 1:
            # Prefill
            position_ids, pad_position = get_position_ids_for_padding_left(
                padding_mask, return_pad_position=True)
        else:
            cur_position = cache_position.value
            pad_position = cur_index - cur_position
            position_ids = cur_position[:, None]
    else:
        position_ids = jnp.arange(num_queries) + cur_index

    position_ids = jnp.broadcast_to(position_ids, (batch_size, num_queries))
    query, key = add_pos(query, key, position_ids)

    indices = (0, cur_index, 0, 0)
    key = lax.dynamic_update_slice(cached_key.value, key, indices)
    value = lax.dynamic_update_slice(cached_value.value, value, indices)
    cached_key.value = key
    cached_value.value = value
    cache_index.value = cache_index.value + num_queries

    if padding_left:
        cache_position.value = position_ids[:, -1] + 1
        padding_mask = jnp.arange(max_length)[None, :] < pad_position[:, None]

    idx = jnp.arange(num_queries) + cur_index
    B = jnp.arange(max_length)[None, :]
    mask = B <= idx[:, None]
    if window_size is not None:
        mask = mask & (B > (idx - window_size)[:, None])
    if padding_mask is not None:
        mask = mask[None, None] & (~padding_mask[:, None, None, :])
    mask = jnp.broadcast_to(
        mask, (batch_size, 1, num_queries, max_length),
    )
    # TODO: maybe use only part of kv for window attention
    return query, key, value, mask


def init_decode_cache(mod, key, value, kv_cache_shard_axes=None):
    is_initialized = mod.has_variable("cache", "cached_key")
    init_fn = jnp.zeros
    if mod.shard and mod.shard_cache:
        init_fn = nn.with_partitioning(init_fn, kv_cache_shard_axes)
    cached_key = mod.variable(
        "cache", "cached_key", init_fn, key.shape, key.dtype
    )
    cached_value = mod.variable(
        "cache", "cached_value", init_fn, value.shape, value.dtype
    )
    cache_index = mod.variable(
        "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
    )
    return is_initialized, cached_key, cached_value, cache_index


def dot_product_attention_weights(
    query: Array,
    key: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    scale: Optional[float] = None,
    attn_logits_soft_cap: Optional[float] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    force_fp32_for_softmax: bool = False,
    einsum_dot_general: Callable[..., Array] = jax.lax.dot_general,
):
    """Computes dot-product attention weights given query and key.

    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.

    Args:
        query: queries for calculating attention with shape of ``[batch...,
            q_length, num_heads, qk_depth_per_head]``.
        key: keys for calculating attention with shape of ``[batch..., kv_length,
            num_heads, qk_depth_per_head]``.
        bias: bias for the attention weights. This should be broadcastable to the
            shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
            incorporating causal masks, padding masks, proximity bias, etc.
        mask: mask for the attention weights. This should be broadcastable to the
            shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
            incorporating causal masks. Attention weights are masked out if their
            corresponding mask value is ``False``.
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        dropout_rng: JAX PRNGKey: to be used for dropout
        dropout_rate: dropout rate
        deterministic: bool, deterministic or not (to apply dropout)
        dtype: the dtype of the computation (default: infer from inputs and params)
        precision: numerical precision of the computation see ``jax.lax.Precision``
            for details.
        force_fp32_for_softmax: bool, whether to force the softmax to be computed in
            fp32. This is useful for mixed-precision training where higher precision
            is desired for numerical stability.
        einsum_dot_general: the dot_general to use in einsum.

    Returns:
        Output of shape ``[batch..., num_heads, q_length, kv_length]``.
    """
    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim, 'q, k must have same rank.'
    assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

    # calculate attention matrix
    depth = query.shape[-1]
    if scale is None:
        scale = 1.0 / jnp.sqrt(depth).astype(dtype)
    else:
        scale = jnp.asarray(scale, dtype=dtype)
    query = query * scale
    # attn weight shape is (batch..., num_heads, q_length, kv_length)

    if query.shape[-2] != key.shape[-2]:
        # grouped query
        num_kv_heads, head_dim = key.shape[-2:]
        query_len, num_heads = query.shape[-3:-1]
        num_groups = num_heads // num_kv_heads
        query = query.reshape(
            *query.shape[:-2], num_kv_heads, num_groups, head_dim)
        attn_weights = jnp.einsum(
            '...qhgd,...khd->...hgqk',
            query,
            key,
            precision=precision,
            _dot_general=einsum_dot_general,
        )
        attn_weights = attn_weights.reshape(
            *attn_weights.shape[:-4], num_kv_heads*num_groups, query_len, -1)
    else:
        attn_weights = jnp.einsum(
            '...qhd,...khd->...hqk',
            query,
            key,
            precision=precision,
            _dot_general=einsum_dot_general,
        )

    if attn_logits_soft_cap is not None:
        attn_logits_soft_cap = jnp.asarray(attn_logits_soft_cap, dtype=dtype)
        attn_weights = jnp.tanh(attn_weights / attn_logits_soft_cap)
        attn_weights = attn_weights * attn_logits_soft_cap

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias
    # apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # normalize the attention weights
    if force_fp32_for_softmax and dtype != jnp.float32:
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32))
    else:
        attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    # apply attention dropout
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # dropout is broadcast across the batch + head dimensions
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    return attn_weights


def replicate_for_multi_query(x, num_heads):
    src_num_heads, head_dim = x.shape[-2:]
    x = jnp.repeat(x, num_heads // src_num_heads, axis=-2)
    return x


# copy from flax.linen.attention.dot_product_attention
def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    scale: Optional[float] = None,
    attn_logits_soft_cap: Optional[float] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    force_fp32_for_softmax: bool = False,
    einsum_dot_general: Callable[..., Array] = jax.lax.dot_general,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    .. note::
    ``query``, ``key``, ``value`` needn't have any batch dimensions.

    Args:
        query: queries for calculating attention with shape of ``[batch...,
            q_length, num_heads, qk_depth_per_head]``.
        key: keys for calculating attention with shape of ``[batch..., kv_length,
            num_heads, qk_depth_per_head]``.
        value: values to be used in attention with shape of ``[batch..., kv_length,
            num_heads, v_depth_per_head]``.
        bias: bias for the attention weights. This should be broadcastable to the
            shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
            incorporating causal masks, padding masks, proximity bias, etc.
        mask: mask for the attention weights. This should be broadcastable to the
            shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
            incorporating causal masks. Attention weights are masked out if their
            corresponding mask value is ``False``.
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        dropout_rng: JAX PRNGKey: to be used for dropout
        dropout_rate: dropout rate
        deterministic: bool, deterministic or not (to apply dropout)
        dtype: the dtype of the computation (default: infer from inputs)
        precision: numerical precision of the computation see ``jax.lax.Precision`
            for details.
        force_fp32_for_softmax: bool, whether to force the softmax to be computed in
            fp32. This is useful for mixed-precision training where higher precision
            is desired for numerical stability.
        einsum_dot_general: the dot_general to use in einsum.

    Returns:
        Output of shape ``[batch..., q_length, num_heads, v_depth_per_head]``.
    """
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), 'q, k, v batch dims must match.'
    assert key.shape[-2] == value.shape[-2], 'k, v num_heads must match.'
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

    # compute attention weights
    attn_weights = dot_product_attention_weights(
        query,
        key,
        bias,
        mask,
        scale,
        attn_logits_soft_cap,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        dtype,
        precision,
        force_fp32_for_softmax,
        einsum_dot_general=einsum_dot_general,
    )
    attn_weights = attn_weights.astype(value.dtype)

    if query.shape[-2] != key.shape[-2]:
        # no significant speed difference between replication and explicit grouped computation

        # If we shard at the last axis, it may be padded to meet 8x128 layout which results 8x expansion of memory.
        # GSPMD automatically recognize it to (X, None, Y, None)
        # num_heads = query.shape[-2]
        # key = replicate_for_multi_query(key, num_heads)
        # value = replicate_for_multi_query(value, num_heads)
        
        num_kv_heads, head_dim = key.shape[-2:]
        num_groups = query.shape[-2] // num_kv_heads
        attn_weights = attn_weights.reshape(
            *attn_weights.shape[:-3], num_kv_heads, num_groups, *attn_weights.shape[-2:])
        attn_outputs = jnp.einsum(
            '...hgqk,...khd->...qhgd',
            attn_weights,
            value,
            precision=precision,
            _dot_general=einsum_dot_general,
        )
        attn_outputs = attn_outputs.reshape(
            *attn_outputs.shape[:-3], num_kv_heads*num_groups, head_dim)
        return attn_outputs
    else:
        return jnp.einsum(
            '...hqk,...khd->...qhd',
            attn_weights,
            value,
            precision=precision,
            _dot_general=einsum_dot_general,
        )


def tpu_flash_attention(
    query, key, value, scale: float | None = None,
    attn_logits_soft_cap: float | None = None,
    is_causal: bool = True, sliding_window_size: int | None = None,
    dtype: Optional[Dtype] = None,
):
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype

    """TPU Flash Attention."""
    # Transpose to ('batch', 'heads', 'length', 'kv')
    query = jnp.transpose(query, axes=(0, 2, 1, 3))
    key = jnp.transpose(key, axes=(0, 2, 1, 3))
    value = jnp.transpose(value, axes=(0, 2, 1, 3))
    if scale is None:
        depth = query.shape[-1]
        scale = 1 / jnp.sqrt(depth)
    scale = jnp.asarray(scale, dtype=query.dtype)
    query = query * scale

    # if decoder_segment_ids is not None:
    #     decoder_segment_ids = splash_attention_kernel.SegmentIds(decoder_segment_ids, decoder_segment_ids)

    def wrap_flash_attention(query, key, value, decoder_segment_ids):
        if decoder_segment_ids is not None:
            assert (
                query.shape[2] == decoder_segment_ids.q.shape[1]
            ), "Sharding along sequence dimension not allowed in tpu kernel attention"
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=min(512, query.shape[2]),
            block_kv_compute=min(512, key.shape[2]),
            block_kv=min(512, key.shape[2]),
            block_q_dkv=min(512, query.shape[2]),
            block_kv_dkv=min(512, key.shape[2]),
            block_kv_dkv_compute=min(512, query.shape[2]),
            block_q_dq=min(512, query.shape[2]),
            block_kv_dq=min(512, query.shape[2]),
        )

        mask = None
        if is_causal:
            mask = splash_attention_mask.CausalMask(shape=(query.shape[2], query.shape[2]))

        if sliding_window_size is not None:
            sliding_mask = splash_attention_mask.LocalMask(
                shape=(query.shape[2], query.shape[2]),
                window_size=(sliding_window_size, sliding_window_size),
                offset=0,
            )
            if mask is not None:
                mask = mask & sliding_mask
            else:
                mask = sliding_mask

        # Create multi-head mask
        multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) *  query.shape[1])
        # splash_attention_kernel.make_splash_mqa not work for gqa
        splash_kernel = splash_attention_kernel.make_splash_mha(
            mask=multi_head_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes, attn_logits_soft_cap=attn_logits_soft_cap,
        )

        return jax.vmap(splash_kernel)(query, key, value, segment_ids=decoder_segment_ids)

    x = wrap_flash_attention(query, key, value, None)
    x = jnp.transpose(x, axes=(0, 2, 1, 3))
    return x.astype(dtype)
