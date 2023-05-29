from jax import lax
from jax import numpy as jnp
import time
import jax
import functools
from flax import linen as nn

from haxllm.model.efficient_attention import dot_product_attention as mefficient_attention

_cur_key = jax.random.PRNGKey(4)


def fresh():
    global _cur_key
    _cur_key, result = jax.random.split(_cur_key)
    return result


batch_size = 16
num_heads = 4
feature_dims = 128
causal = True


def fresh_qkv(size, dtype=jnp.bfloat16):
    qkv_shape = (batch_size, size, num_heads, feature_dims)
    return jax.random.normal(fresh(), qkv_shape, dtype=dtype)


def fresh_mask(size):
    return nn.make_causal_mask(jnp.ones((batch_size, size), dtype=jnp.bool_), dtype=jnp.bool_)


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
execute_memory_efficient_att = True
execute_memory_efficient_att2 = True
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
    if causal:
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

    if execute_memory_efficient_att:
        mefficient_attn = functools.partial(
            mefficient_attention, causal=causal, precision=precision, dtype=dtype
        )

        compilation_start = time.time()
        compilation_res = mefficient_attn(query, key, value)
        compilation_res.block_until_ready()
        print(
            "Memory-efficient attention compilation time:",
            time.time() - compilation_start,
        )

        total_time_mem = 0.0
        for _ in range(repeats):
            start = time.time()
            res = mefficient_attn(query, key, value)
            res.block_until_ready()
            total_time_mem += time.time() - start
        total_time_mem = total_time_mem / repeats
        print("Memory-efficient attention took:", total_time_mem)

    if execute_memory_efficient_att2:
        mefficient_attn2 = functools.partial(
            mefficient_attention, causal=causal, sparse=True, precision=precision, dtype=dtype,
            query_chunk_size=1024, key_chunk_size=1024,
        )

        compilation_start = time.time()
        compilation_res = mefficient_attn2(query, key, value)
        compilation_res.block_until_ready()
        print(
            "Memory-efficient attention2 compilation time:",
            time.time() - compilation_start,
        )

        total_time_mem2 = 0.0
        for _ in range(repeats):
            start = time.time()
            res2 = mefficient_attn2(query, key, value)
            res2.block_until_ready()
            total_time_mem2 += time.time() - start
        total_time_mem2 = total_time_mem2 / repeats
        print("Memory-efficient attention2 took:", total_time_mem2)

    if execute_standard_att and execute_memory_efficient_att:
        print("Performance advantage:", (total_time / total_time_mem) - 1.0)
        diff = res - res_std
        print("avg difference", jnp.average(jnp.abs(diff)))
        print("max difference", jnp.max(jnp.abs(diff)))
        # np.testing.assert_allclose(
        #     res.astype(jnp.float32), res_std.astype(jnp.float32), atol=2e-2
        # )

    if execute_memory_efficient_att and execute_memory_efficient_att2:
        print("Performance advantage:", (total_time_mem / total_time_mem2) - 1.0)
        diff = res2 - res
        print("avg difference", jnp.average(jnp.abs(diff)))
        print("max difference", jnp.max(jnp.abs(diff)))
        # np.testing.assert_allclose(
        #     res2.astype(jnp.float32), res.astype(jnp.float32), atol=2e-2
        # )


# Compare differentiation performance

execute_standard_att = True
execute_memory_efficient_att = True
execute_memory_efficient_att2 = True

input_dtype = jnp.bfloat16
dtype = jnp.bfloat16
precision = lax.Precision.DEFAULT


def loss_simp(query, key, value, mask):
    return jnp.sum(standard_attention(query, key, value, mask, precision=precision))

def loss_ckpt(query, key, value):
    return jnp.sum(mefficient_attention(query, key, value, causal=causal, precision=precision))

def loss_ckpt2(query, key, value):
    return jnp.sum(mefficient_attention(query, key, value, causal=causal, sparse=True, precision=precision,
                                        query_chunk_size=1024, key_chunk_size=1024))


diff_attention_simp = jax.jit(jax.grad(loss_simp, argnums=[0, 1, 2]))
diff_mefficient_attention = jax.jit(jax.grad(loss_ckpt, argnums=[0, 1, 2]))
diff_mefficient_attention2 = jax.jit(jax.grad(loss_ckpt2, argnums=[0, 1, 2]))

for i in range(8, 12, 1):
    q_size = 2**i
    memsize = 2**i
    print("\nAttention size:", q_size, "x", memsize)

    query = fresh_qkv(q_size, input_dtype)
    key = fresh_qkv(memsize, input_dtype)
    value = fresh_qkv(memsize, input_dtype)
    mask = fresh_mask(q_size)

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

    if execute_memory_efficient_att:
        compilation_start = time.time()
        _comp_res = diff_mefficient_attention(query, key, value)
        for t in _comp_res:
            t.block_until_ready()
        print("Diff mem ckpt compilation time:", time.time() - compilation_start)

        total_time_mem = 0.0
        for _ in range(repeats):
            start = time.time()
            res = diff_mefficient_attention(query, key, value)
            for t in res:
                t.block_until_ready()
            total_time_mem += time.time() - start
        total_time_mem = total_time_mem / repeats
        print("Memory-efficient attention took:", total_time_mem)

    if execute_memory_efficient_att2:
        compilation_start = time.time()
        _comp_res = diff_mefficient_attention2(query, key, value)
        for t in _comp_res:
            t.block_until_ready()
        print("Diff mem2 ckpt compilation time:", time.time() - compilation_start)

        total_time_mem2 = 0.0
        for _ in range(repeats):
            start = time.time()
            res2 = diff_mefficient_attention2(query, key, value)
            for t in res2:
                t.block_until_ready()
            total_time_mem2 += time.time() - start
        total_time_mem2 = total_time_mem2 / repeats
        print("Memory-efficient attention2 took:", total_time_mem2)

    if execute_standard_att and execute_memory_efficient_att:
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

    if execute_memory_efficient_att and execute_memory_efficient_att2:
        print("Performance advantage:", (total_time_mem / total_time_mem2) - 1.0)
        diff = res2[0] - res[0]
        print("avg difference", jnp.average(jnp.abs(diff)))
        print("max difference", jnp.max(jnp.abs(diff)))
        # for tuple_idx in range(3):
        #     np.testing.assert_allclose(
        #         res2[tuple_idx].astype(jnp.float32),
        #         res[tuple_idx].astype(jnp.float32),
        #         atol=3e-2,
        #         rtol=1e-2,
        #     )