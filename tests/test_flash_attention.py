from jax import lax
from jax import numpy as jnp
import time
import jax
import functools

from haxllm.model.attention import dot_product_attention, tpu_flash_attention
from jax_smi import initialise_tracking
initialise_tracking()


_cur_key = jax.random.PRNGKey(4)


def fresh():
    global _cur_key
    _cur_key, result = jax.random.split(_cur_key)
    return result


batch_size = 16
num_groups = 16
num_heads = 32
head_dim = 128
decode = False  # not supported in flash attention
mask_mode = 'causal'


def fresh_q(size, dtype=jnp.bfloat16):
    q_shape = (batch_size, size, num_heads, head_dim)
    return jax.random.normal(fresh(), q_shape, dtype=dtype)

def fresh_kv(size, dtype=jnp.bfloat16):
    kv_shape = (batch_size, size, num_heads // num_groups, head_dim)
    return jax.random.normal(fresh(), kv_shape, dtype=dtype)

def fresh_mask(q_len, kv_len):
    # broadcasted to (batch_size, num_heads, q_length, kv_length)
    idxs1 = jnp.arange(q_len, dtype=jnp.int32)
    idxs2 = jnp.arange(kv_len, dtype=jnp.int32)
    idxs1 = idxs1[None, :, None]
    idxs2 = idxs2[None, None, :]
    mask = (idxs1 >= idxs2)
    mask = mask[:, None, :, :]  # (batch_size, 1, seq_len, seq_len)
    return mask

def standard_attention(query, key, value, mask=None, dtype=jnp.float32, precision=None):
    return dot_product_attention(
        query, key, value, mask=mask, dtype=dtype, precision=precision, force_fp32_for_softmax=True)

def flash_attention(query, key, value, dtype):
    return tpu_flash_attention(query, key, value, is_causal=True, dtype=dtype)

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

for i in range(8, 13, 1):
    q_size = 2**i if not decode else 1
    memsize = 2**i
    print("\nAttention size:", q_size, "x", memsize)
    query, key, value = (
        fresh_q(q_size, input_dtype),
        fresh_kv(memsize, input_dtype),
        fresh_kv(memsize, input_dtype),
    )
    if mask_mode:
        mask = fresh_mask(q_size, memsize)
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
            flash_attention, dtype=dtype
        )
        flash_attn = jax.jit(_orig_attn2)

        compilation_start = time.time()
        compilation_res = flash_attn(query, key, value)
        compilation_res.block_until_ready()
        print(
            "Flash attention compilation time:",
            time.time() - compilation_start,
        )

        total_time_mem = 0.0
        for _ in range(repeats):
            start = time.time()
            res = flash_attn(query, key, value)
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

if decode:
    raise RuntimeError("Stpo")

# Compare differentiation performance

execute_standard_att = True
execute_flash_att = False

def loss_simp(query, key, value, mask):
    return jnp.sum(standard_attention(query, key, value, mask, dtype=dtype, precision=precision))

def loss_ckpt(query, key, value, mask):
    return jnp.sum(flash_attention(query, key, value, dtype=dtype))


diff_attention_simp = jax.jit(jax.grad(loss_simp, argnums=[0, 1, 2]))
diff_flash_attention = jax.jit(jax.grad(loss_ckpt, argnums=[0, 1, 2]))

for i in range(8, 13, 1):
    q_size = 2**i
    memsize = 2**i
    print("\nAttention size:", q_size, "x", memsize)

    query = fresh_q(q_size, input_dtype)
    key = fresh_kv(memsize, input_dtype)
    value = fresh_kv(memsize, input_dtype)
    if mask_mode:
        mask = fresh_mask(q_size, memsize)
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
