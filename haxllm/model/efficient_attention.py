import functools, jax, math
from jax import lax
from jax import numpy as jnp


def make_causal_mask(row_start, row_size, col_start, col_size):
    r"""
    Make a mask for a chunk of queries against a chunk of keys.

    Returns
    -------
    mask: jnp.ndarray, shape (row_size, 1, col_size)
    
    """
    mask = (row_start + jnp.arange(0, row_size)[:, None]) >= \
        (col_start + jnp.arange(0, col_size)[None, :])
    return jnp.expand_dims(mask, axis=1)


def make_padding_mask(row_start, row_size, col_start, col_size, pad_position):
    pad_position = jnp.where(pad_position == -1, col_size, pad_position)
    mask = col_start + jnp.arange(0, col_size) < pad_position
    mask = jnp.expand_dims(mask, axis=(0, 1))
    return mask


def make_bidirectional_mask(row_start, row_size, col_start, col_size, context_length):
    mask1 = col_start + jnp.arange(0, col_size) < context_length
    mask2 = (row_start + jnp.arange(0, row_size)[:, None]) >= \
        (col_start + jnp.arange(0, col_size)[None, :])
    mask = mask1[None, :] | mask2
    mask = jnp.expand_dims(mask, axis=1)
    return mask


def _query_chunk_attention(
    query,
    key,
    value,
    query_chunk_idx,
    pad_position=None,
    context_length=None,
    mask_mode='causal',
    sparse=False,
    key_chunk_size=4096,
    precision=None,
    dtype=jnp.float32,
):
    query_chunk_size = query.shape[0]
    num_kv, num_heads, k_features = key.shape
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features).astype(dtype)
    big_neg = jnp.finfo(dtype).min

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value, chunk_idx):
        attn_weights = jnp.einsum(
            "qhd,khd->qhk", query, key, precision=precision
        ).astype(dtype)
        if mask_mode == 'causal':
            mask = make_causal_mask(
                query_chunk_idx, query_chunk_size, chunk_idx, key_chunk_size
            )
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        elif mask_mode == 'padding':
            mask = make_padding_mask(
                query_chunk_idx, query_chunk_size, chunk_idx, key_chunk_size, pad_position
            )
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        elif mask_mode == 'bidirectional':
            mask = make_bidirectional_mask(
                query_chunk_idx, query_chunk_size, chunk_idx, key_chunk_size, context_length
            )
            attn_weights = jnp.where(mask, attn_weights, big_neg)
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum(
            "vhf,qhv->qhf", value, exp_weights, precision=precision
        ).astype(dtype)
        return (
            exp_values,
            exp_weights.sum(axis=-1),
            max_score.reshape((query.shape[0], num_heads)),
        )

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk2(query, key, value, chunk_idx):
        max_score = jnp.full((query_chunk_size, num_heads, 1), big_neg, dtype=dtype)
        exp_weights = jnp.ones((query_chunk_size, num_heads, key_chunk_size), dtype=dtype)
        exp_values = jnp.einsum(
            "vhf,qhv->qhf", value, exp_weights, precision=precision
        ).astype(dtype)
        return (
            exp_values,
            exp_weights.sum(axis=-1),
            max_score.reshape((query.shape[0], num_heads)),
        )
    
    def chunk_scanner(chunk_idx):
        key_chunk = lax.dynamic_slice(
            key, (chunk_idx, 0, 0), slice_sizes=(key_chunk_size, num_heads, k_features)
        )
        value_chunk = lax.dynamic_slice(
            value,
            (chunk_idx, 0, 0),
            slice_sizes=(key_chunk_size, num_heads, v_features),
        )
        if sparse:
            row_end = query_chunk_idx + query_chunk_size
            col_start = chunk_idx
            return lax.cond(
                row_end >= col_start,
                summarize_chunk, summarize_chunk2,
                query, key_chunk, value_chunk, chunk_idx
            )
        else:
            return summarize_chunk(query, key_chunk, value_chunk, chunk_idx)

    chunk_values, chunk_weights, chunk_max = lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size)
    )

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights


def _mefficient_attention(
    query,
    key,
    value,
    pad_position=None,
    context_length=None,
    mask_mode='causal',
    sparse=False,
    query_chunk_size=1024,
    key_chunk_size=4096,
    precision=None,
    dtype=jnp.float32,
):
    num_q, num_heads, q_features = query.shape

    def chunk_scanner(chunk_idx, _):
        query_chunk = lax.dynamic_slice(
            query,
            (chunk_idx, 0, 0),
            slice_sizes=(min(query_chunk_size, num_q), num_heads, q_features),
        )
        return (
            chunk_idx + query_chunk_size,
            _query_chunk_attention(
                query_chunk, key, value, chunk_idx,
                pad_position, context_length,
                mask_mode=mask_mode, sparse=sparse,
                key_chunk_size=key_chunk_size,
                precision=precision, dtype=dtype
            ),
        )

    _, res = lax.scan(
        chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size)
    )
    return res.reshape(num_q, num_heads, value.shape[-1])


# TODO: add support for key_padding_mask
@functools.partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10))
def dot_product_attention(
    query,
    key,
    value,
    pad_position=None,
    context_length=None,
    mask_mode='causal',
    sparse=False,
    query_chunk_size=1024,
    key_chunk_size=4096,
    precision=None,
    dtype=jnp.float32,
):
    assert mask_mode in ['causal', 'padding', 'bidirectional', 'none']
    if mask_mode == 'padding':
        assert pad_position is not None
    elif mask_mode == 'bidirectional':
        assert context_length is not None
    fun = functools.partial(
        _mefficient_attention, mask_mode=mask_mode, sparse=sparse,
        query_chunk_size=query_chunk_size, key_chunk_size=key_chunk_size,
        precision=precision, dtype=dtype)
    return jax.vmap(fun, in_axes=(0, 0, 0, 0, 0))(query, key, value, pad_position, context_length)
