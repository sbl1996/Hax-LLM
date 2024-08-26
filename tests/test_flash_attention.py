import math
from jax import lax
from jax import numpy as jnp
import time
import jax
import functools

from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as flash_attention_tpu, BlockSizes

_cur_key = jax.random.PRNGKey(4)


def fresh():
    global _cur_key
    _cur_key, result = jax.random.split(_cur_key)
    return result

def get_block_sizes(q_len, kv_len):
    return BlockSizes(
            block_q=min(q_len, 128),
            block_k_major=min(kv_len, 1024),
            block_k=min(kv_len, 1024),
            block_b=1,
            block_q_major_dkv=128,
            block_k_major_dkv=128,
            block_k_dkv=128,
            block_q_dkv=128,
            block_k_major_dq=128,
            block_k_dq=128,
            block_q_dq=128,
        )

batch_size = 16
num_heads = 32
feature_dims = 128
mask_mode = 'bidirectional'
context_length = jax.random.randint(fresh(), (batch_size,), 0, 64)


def fresh_qkv(size, dtype=jnp.bfloat16):
    qkv_shape = (batch_size, size, num_heads, feature_dims)
    return jax.random.normal(fresh(), qkv_shape, dtype=dtype)


def fresh_mask(size):
    # broadcasted to (batch_size, num_heads, q_length, kv_length)
    # mask = jnp.arange(size)[None, :] < pad_position[:, None]
    # return jnp.broadcast_to(mask[:, None, None, :], (batch_size, 1, size, size))
    # return nn.make_causal_mask(jnp.ones((batch_size, size), dtype=jnp.bool_), dtype=jnp.bool_)
    idxs = jnp.arange(size, dtype=jnp.int32)
    idxs1 = idxs[None, :, None]
    idxs2 = idxs[None, None, :]
    mask = (idxs1 >= idxs2) | (idxs2 < context_length[:, None, None])
    mask = mask[:, None, :, :]  # (batch_size, 1, seq_len, seq_len)
    return mask


def flash_attention(query, key, value, mask=None, dtype=jnp.float32, precision=None):
    dtype = query.dtype
    q, k, v = jax.tree.map(lambda x: jnp.swapaxes(x.astype(dtype), 1, 2), (query, key, value))
    depth = query.shape[-1]
    sm_scale = 1.0 / math.sqrt(depth)
    if mask is not None:
        ab = jnp.where(mask, 0.0, jnp.finfo(dtype).min)
    else:
        ab = None
    ab = jnp.broadcast_to(ab, (batch_size, num_heads, query.shape[1], key.shape[1]))
    block_sizes = get_block_sizes(q.shape[2], k.shape[2])
    o = flash_attention_tpu(q, k, v, sm_scale=sm_scale, ab=ab, block_sizes=block_sizes)
    o = jnp.swapaxes(o, 1, 2).astype(dtype)
    return o


def standard_attention(query, key, value, mask=None, dtype=jnp.float32, precision=None):
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum("...qhd,...khd->...hqk", query, key, precision=precision)
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # normalize the attention weights
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)
    return jnp.einsum("...hqk,...khd->...qhd", attn_weights, value, precision=precision)


# Compare inference performance

# The evaluation uses mixed-precision by default (bfloat16 for the inputs and
# outputs, and float32 for certain internal representations.)
input_dtype = jnp.bfloat16
dtype = jnp.bfloat16
# Using HIGHEST means that we use full float32 precision. For neural network
# training we can often use lax.Precision.DEFAULT instead.
precision = lax.Precision.DEFAULT

execute_standard_att = True
execute_flash_att = True
repeats = 200

for i in range(8, 12, 1):
    q_size = 2**i
    memsize = 2**i
    print("\nAttention size:", q_size, "x", memsize)
    query, key, value = (
        fresh_qkv(q_size, input_dtype),
        fresh_qkv(memsize, input_dtype),
        fresh_qkv(memsize, input_dtype),
    )
    if mask_mode:
        mask = fresh_mask(q_size)
    else:
        mask = None

    if execute_standard_att:
        _orig_attn = functools.partial(
            standard_attention, precision=precision, dtype=dtype
        )
        standard_attn = jax.jit(_orig_attn)

        compilation_start = time.time()
        compilation_res = standard_attn(query, key, value, mask)
        compilation_res.block_until_ready()
        print("Standard compilation time:", time.time() - compilation_start)

        total_time = 0.0
        for _ in range(repeats):
            start = time.time()
            res_std = standard_attn(query, key, value, mask)
            res_std.block_until_ready()
            # print('Time of op:', time.time() - start)
            total_time += time.time() - start
        total_time = total_time / repeats
        print("Standard attention took:", total_time)

    if execute_flash_att:
        _orig_attn2 = functools.partial(
            flash_attention, precision=precision, dtype=dtype
        )
        flash_attn = jax.jit(_orig_attn2)

        compilation_start = time.time()
        compilation_res = flash_attn(query, key, value, mask)
        compilation_res.block_until_ready()
        print(
            "Flash attention compilation time:",
            time.time() - compilation_start,
        )

        total_time_mem = 0.0
        for _ in range(repeats):
            start = time.time()
            res = flash_attn(query, key, value, mask)
            res.block_until_ready()
            total_time_mem += time.time() - start
        total_time_mem = total_time_mem / repeats
        print("Flash attention took:", total_time_mem)

    if execute_standard_att and execute_flash_att:
        print("Performance advantage:", (total_time / total_time_mem) - 1.0)
        diff = res - res_std
        print("avg difference", jnp.average(jnp.abs(diff)))
        print("max difference", jnp.max(jnp.abs(diff)))
        # np.testing.assert_allclose(
        #     res.astype(jnp.float32), res_std.astype(jnp.float32), atol=2e-2
        # )

# Compare differentiation performance

execute_standard_att = True
execute_flash_att = True

def loss_simp(query, key, value, mask):
    return jnp.sum(standard_attention(query, key, value, mask, dtype=dtype, precision=precision))

def loss_ckpt(query, key, value, mask):
    return jnp.sum(flash_attention(query, key, value, mask, dtype=dtype, precision=precision))


diff_attention_simp = jax.jit(jax.grad(loss_simp, argnums=[0, 1, 2]))
diff_flash_attention = jax.jit(jax.grad(loss_ckpt, argnums=[0, 1, 2]))

for i in range(8, 12, 1):
    q_size = 2**i
    memsize = 2**i
    print("\nAttention size:", q_size, "x", memsize)

    query = fresh_qkv(q_size, input_dtype)
    key = fresh_qkv(memsize, input_dtype)
    value = fresh_qkv(memsize, input_dtype)
    if mask_mode:
        mask = fresh_mask(q_size)
    else:
        mask = None

    if execute_standard_att:
        compilation_start = time.time()
        _comp_res = diff_attention_simp(query, key, value, mask)
        for t in _comp_res:
            t.block_until_ready()
        print("Diff simp compilation time:", time.time() - compilation_start)

        total_time_simp = 0.0
        for _ in range(repeats):
            start = time.time()
            res_std = diff_attention_simp(query, key, value, mask)
            for t in res_std:
                t.block_until_ready()
            total_time_simp += time.time() - start
        total_time_simp = total_time_simp / repeats
        print("Standard attention took:", total_time_simp)

    if execute_flash_att:
        compilation_start = time.time()
        _comp_res = diff_flash_attention(query, key, value, mask)
        for t in _comp_res:
            t.block_until_ready()
        print("Diff mem ckpt compilation time:", time.time() - compilation_start)

        total_time_mem = 0.0
        for _ in range(repeats):
            start = time.time()
            res = diff_flash_attention(query, key, value, mask)
            for t in res:
                t.block_until_ready()
            total_time_mem += time.time() - start
        total_time_mem = total_time_mem / repeats
        print("Flash attention took:", total_time_mem)

    if execute_standard_att and execute_flash_att:
        print("Performance advantage:", (total_time_simp / total_time_mem) - 1.0)
        diff = res[0] - res_std[0]
        print("avg difference", jnp.average(jnp.abs(diff)))
        print("max difference", jnp.max(jnp.abs(diff)))
        # for tuple_idx in range(3):
        #     np.testing.assert_allclose(
        #         res[tuple_idx].astype(jnp.float32),
        #         res_std[tuple_idx].astype(jnp.float32),
        #         atol=2e-2,
        #         rtol=1e-2,
        #     )
