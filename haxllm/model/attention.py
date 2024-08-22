from typing import Optional
import math
import numpy as np

import jax.numpy as jnp
from jax import lax

import flax.linen as nn
from haxllm.model.mixin import RoPEScalingConfig


def compute_inv_freq(dim: int, base = 10000.0):
    return 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32)[: (dim // 2)] / dim))


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
    t = np.arange(end, dtype=np.float32)  # type: ignore
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


# from chatglm2, different from original rope
def precompute_freqs_cis2(inv_freq, end: int, dtype = jnp.float32):
    # returns:
    #   cos, sin: (end, dim)
    freqs = inv_freq
    t = np.arange(end, dtype=np.float32)  # type: ignore
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
    if rope_type == "chatglm2":
        inv_freq = compute_inv_freq(head_dim // 2, base)
        cos, sin = precompute_freqs_cis2(inv_freq, max_len, dtype=dtype)
        add_pos = lambda q, k, p=None: apply_rotary_pos_emb_index2(q, k, cos, sin, p)
    elif rope_type == "default":
        inv_freq = compute_inv_freq(head_dim, base)
        cos, sin = precompute_freqs_cis(inv_freq, max_len, dtype=dtype)
        add_pos = lambda q, k, p=None: apply_rotary_pos_emb_index(q, k, cos, sin, p)
    elif rope_type == "llama3":
        inv_freq = compute_llama3_inv_freq(
            head_dim, base, scaling.factor, scaling.low_freq_factor, scaling.high_freq_factor,
            scaling.max_position_embeddings)
        cos, sin = precompute_freqs_cis(inv_freq, max_len, dtype=dtype)
        add_pos = lambda q, k, p=None: apply_rotary_pos_emb_index(q, k, cos, sin, p)
    elif rope_type == "dynamic":
        orig_max_len = scaling.max_position_embeddings
        if max_len > orig_max_len:
            alpha = scaling.factor * max_len / orig_max_len - (scaling.factor - 1)
            base = base * alpha ** (dim / (dim - 2))
        inv_freq = compute_inv_freq(head_dim, base)
        cos, sin = precompute_freqs_cis(inv_freq, max_len, dtype=dtype)
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


def decode_for_padding_left(add_pos, query, key, value, cache_index, cached_key, cached_value, cache_position, padding_mask=None):
    num_queries = query.shape[-3]
    cur_index = cache_index.value

    assert key.ndim == 4 and value.ndim == 4, "Only 1D batched input is supported for decoding."
    batch_size, max_length, num_heads, depth_per_head = cached_key.value.shape

    if num_queries > 1:
        # First stage, context decode
        position_ids, pad_position = get_position_ids_for_padding_left(
            padding_mask, return_pad_position=True)
    else:
        cur_position = cache_position.value
        pad_position = cur_index - cur_position
        position_ids = cur_position[:, None]
    position_ids = jnp.broadcast_to(position_ids, (batch_size, num_queries))
    query, key = add_pos(query, key, position_ids)

    indices = (0, cur_index, 0, 0)
    key = lax.dynamic_update_slice(cached_key.value, key, indices)
    value = lax.dynamic_update_slice(cached_value.value, value, indices)
    cached_key.value = key
    cached_value.value = value

    cache_position.value = position_ids[:, -1] + 1
    cache_index.value = cache_index.value + num_queries

    padding_mask = jnp.arange(max_length)[None, :] < pad_position[:, None]

    idx = jnp.arange(num_queries) + cur_index
    mask = jnp.arange(max_length)[None, :] <= idx[:, None]
    mask = mask[None, None] & (~padding_mask[:, None, None, :])
    mask = jnp.broadcast_to(
        mask, (batch_size, 1, num_queries, max_length),
    )
    return query, key, value, mask


def decode_for_padding_right(add_pos, query, key, value, cache_index, cached_key, cached_value):
    num_queries = query.shape[-3]
    cur_index = cache_index.value

    *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape

    position_ids = jnp.arange(num_queries) + cur_index
    position_ids = jnp.broadcast_to(position_ids, tuple(batch_dims) + (num_queries,))
    query, key = add_pos(query, key, position_ids)

    indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
    key = lax.dynamic_update_slice(cached_key.value, key, indices)
    value = lax.dynamic_update_slice(cached_value.value, value, indices)
    cached_key.value = key
    cached_value.value = value

    cache_index.value = cache_index.value + num_queries

    idx = jnp.arange(num_queries) + cur_index
    mask = jnp.arange(max_length)[None, :] <= idx[:, None]
    mask = jnp.broadcast_to(
        mask, tuple(batch_dims) + (1, num_queries, max_length),
    )
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