from functools import partial

import jax
import jax.numpy as jnp

from flax.core.frozen_dict import unfreeze, FrozenDict


def add_batch_dim(x, batch_size):
    if x.ndim <= 2:
        # cache_index and cache_position_ids
        return jnp.tile(x, [1] * x.ndim)
    reps = [1] * x.ndim
    reps[-4] = batch_size
    return jnp.tile(x, reps)


@jax.jit
def gather_beams(xs, indices):
    def gather_fn(x):
        if x.ndim <= 2:
            return x
        return x[..., indices, :, :, :]
    return jax.tree_map(gather_fn, xs)


def fix_cache_index(cache, offset):
    if isinstance(cache, FrozenDict):
        cache = unfreeze(cache)
    if 'hs' in cache['transformer']:
        v = cache['transformer']['hs']['attn']['cache_index']
        cache['transformer']['hs']['attn']['cache_index'] = v - offset
    else:
        keys = [ k for k in cache['transformer'] if k.startswith('h_') ]
        for key in keys:
            v = cache['transformer'][key]['attn']['cache_index']
            cache['transformer'][key]['attn']['cache_index'] = v - offset
    return cache


@jax.jit
def sample_token_top_p_single(logits, rng, p):
    sorted_indices = jnp.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))
    mask = cumulative_probs > p
    mask = jnp.concatenate([jnp.zeros((1,), dtype=jnp.bool_), mask[:-1]])
    sorted_logits = jnp.where(mask, -float('Inf'), sorted_logits)
    token = jax.random.categorical(rng, sorted_logits)
    token = sorted_indices[token]
    return token


@jax.jit
def sample_token_top_p(logits, rng, p):
    if logits.ndim == 1:
        return sample_token_top_p_single(logits, rng, p)
    elif logits.ndim == 2:
        if rng.ndim == 2:
            assert logits.shape[0] == rng.shape[0]
            return jax.vmap(sample_token_top_p_single, in_axes=(0, 0, None))(logits, rng, p)
        else:
            return jax.vmap(sample_token_top_p_single, in_axes=(0, None, None))(logits, rng, p)
    else:
        raise NotImplementedError


@partial(jax.jit, static_argnums=(2,))
def sample_token_top_k_single(logits, rng, k):
    logits, tokens = jax.lax.top_k(logits, k)
    index = jax.random.categorical(rng, logits)
    token = tokens[index]
    return token


@partial(jax.jit, static_argnums=(2,))
def sample_token_top_k(logits, rng, k):
    if logits.ndim == 1:
        return sample_token_top_k_single(logits, rng, k)
    elif logits.ndim == 2:
        if rng.ndim == 2:
            assert logits.shape[0] == rng.shape[0]
            return jax.vmap(sample_token_top_k_single, in_axes=(0, 0, None))(logits, rng, k)
        else:
            return jax.vmap(sample_token_top_k_single, in_axes=(0, None, None))(logits, rng, k)
    else:
        raise NotImplementedError


def sample_token(logits, rng, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1):
    if temperature < 1e-5 or top_k == 1:
        return jnp.argmax(logits, axis=-1)
    logits = logits / temperature
    if top_k > 1:
        return sample_token_top_k(logits, rng, top_k)
    elif top_p < 1.0:
        return sample_token_top_p(logits, rng, top_p)
    else:
        if logits.ndim == 2:
            if rng.ndim == 2:
                assert logits.shape[0] == rng.shape[0]
                return jax.vmap(jax.random.categorical, in_axes=(0, 0))(rng, logits)
            else:
                return jax.random.categorical(rng, logits, axis=-1)
        else:
            return jax.random.categorical(rng, logits, axis=-1)


def split_rng(rng):
    if rng is None:
        return None, None
    return jax.random.split(rng)


def random_sample(inputs, tokenizer, apply_fn, params, cache, max_len,
                  temperature=1.0, top_k=5, top_p=1.0, rng=None, two_stage=False, pad_context=None,
                  pad_token_id=None, decode=True, stop_token_ids=None):
    if stop_token_ids is None:
        stop_token_ids = [tokenizer.eos_token_id]
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)
    if temperature < 1e-5:
        top_k = 1
    if temperature > 1e-5 or top_k != 1:
        assert rng is not None, "Must provide rng if not using greedy decoding"
    if pad_context is not None:
        assert two_stage, "Must use two_stage if pad_context is not None"
        assert pad_token_id is not None, "Must provide pad_token_id if pad_context is not None"
    if isinstance(inputs, str):
        tokens = tokenizer(inputs)['input_ids']
    else:
        tokens = inputs
    context_length = len(tokens)
    assert context_length < max_len, "Context length must be less than max_len"

    cache = jax.tree_map(lambda x: jnp.tile(x, [1] * x.ndim), cache)

    if pad_context is None:
        live_seq = jnp.zeros((1, max_len), dtype=jnp.int32)
    else:
        live_seq = jnp.full((1, max_len), pad_token_id, dtype=jnp.int32)
    live_seq = live_seq.at[:, :context_length].set(tokens)

    if two_stage:
        first_len = context_length if pad_context is None else pad_context
        input_ids = live_seq[:, :first_len]
        cache, logits = apply_fn(params, cache, input_ids)
        logits = logits.astype(jnp.float32)[0, context_length-1]
        rng, subrng = split_rng(rng)
        token = sample_token(logits, subrng, temperature, top_p, top_k)
        live_seq = live_seq.at[:, context_length].set(token)
        if pad_context is not None:
            cache = fix_cache_index(cache, pad_context - context_length)
        i = context_length
    else:
        i = 0

    while i < max_len:
        input_ids = live_seq[:, [i]]
        cache, logits = apply_fn(params, cache, input_ids)
        i += 1
        if i < context_length:
            continue
        logits = logits.astype(jnp.float32)[0, 0]
        rng, subrng = split_rng(rng)
        token = sample_token(logits, subrng, temperature, top_p, top_k)
        live_seq = live_seq.at[:, i].set(token)
        if token in stop_token_ids:
            break
    live_seq = jax.device_get(live_seq[0, :i])
    if decode:
        return tokenizer.decode(live_seq)
    else:
        return live_seq


def batch_random_sample(
    input_ids, apply_fn, params, cache, max_len, temperature=1.0, top_k=5, top_p=1.0,
    rng=None, pad_token_id=None, eos_token_id=None, stop_token_ids=None):
    # TODO: using right padding to batch first stage context encode
    if stop_token_ids is None:
        stop_token_ids = [eos_token_id]
    if eos_token_id not in stop_token_ids:
        stop_token_ids.append(eos_token_id)
    if temperature < 1e-5:
        top_k = 1
    if temperature >= 1e-5 or top_k != 1:
        assert rng is not None, "Must provide rng if not using greedy decoding"
    assert pad_token_id is not None and eos_token_id is not None, "Must provide pad_token_id and eos_token_id" 
    # (batch_size, max_source_length)
    batch_size, max_source_length = input_ids.shape
    context_length = jnp.argmax(input_ids == pad_token_id, axis=1)
    context_length = jnp.where(context_length == 0, max_source_length, context_length)

    cache = jax.tree_map(lambda x: add_batch_dim(x, batch_size), cache)

    live_seqs = jnp.full((batch_size, max_len), pad_token_id, dtype=jnp.int32)
    live_seqs = live_seqs.at[:, :max_source_length].set(input_ids)

    i = 0
    end_index = jnp.zeros((batch_size,), dtype=jnp.int32)
    is_end = jnp.zeros((batch_size,), dtype=jnp.bool_)
    while i < max_len:
        input_ids = live_seqs[:, [i]]
        cache, logits = apply_fn(params, cache, input_ids)
        i += 1
        logits = logits.astype(jnp.float32)[:, 0]
        if rng is None:
            subrngs = None
        else:
            rngs = jax.random.split(rng, batch_size + 1)
            rng, subrngs = rngs[0], rngs[1:]
        tokens = sample_token(logits, subrngs, temperature, top_p, top_k)
        is_context = i < context_length
        tokens = jnp.where(
            is_end | is_context, live_seqs[:, i], tokens)
        live_seqs = live_seqs.at[:, i].set(tokens)

        is_stop_token = tokens == stop_token_ids[0]
        for stop_token_id in stop_token_ids[1:]:
            is_stop_token = is_stop_token | (tokens == stop_token_id)
        is_end = is_end | (~is_context & is_stop_token)
        end_index = jnp.where(
            is_end & (end_index == 0),
            jnp.full_like(end_index, i),
            end_index,
        )
        if jnp.all(is_end):
            break
    return jax.device_get(live_seqs[:, :i])


def beam_search(inputs, tokenizer, apply_fn, params, cache, max_len,
                n_beams, two_stage=False, pad_context=None, pad_token_id=None):
    if pad_context is not None:
        assert two_stage, "Must use two_stage if pad_context is not None"
        assert pad_token_id is not None, "Must provide pad_token_id if pad_context is not None"
    if isinstance(inputs, str):
        tokens = tokenizer(inputs)['input_ids']
    else:
        tokens = inputs
    context_length = len(tokens)
    assert context_length < max_len, "Context length must be less than max_len"

    cache = jax.tree_map(lambda x: add_batch_dim(x, n_beams), cache)

    if pad_context is None:
        live_seqs = jnp.zeros((n_beams, max_len), dtype=jnp.int32)
    else:
        live_seqs = jnp.full((n_beams, max_len), pad_token_id, dtype=jnp.int32)
    live_seqs = live_seqs.at[:, :context_length].set(tokens)
    if two_stage:
        first_len = context_length if pad_context is None else pad_context
        input_ids = live_seqs[:, :first_len]
        cache, logits = apply_fn(params, cache, input_ids)
        if pad_context is not None:
            cache = fix_cache_index(cache, pad_context - context_length)
    else:
        for i in range(context_length):
            input_ids = live_seqs[:, [i]]
            cache, logits = apply_fn(params, cache, input_ids)
    i = context_length
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits[0, i-1], axis=0)
    live_log_probs, topk_indices = jax.lax.top_k(log_probs, n_beams)
    live_seqs = live_seqs.at[:, i].set(topk_indices)

    vocab_size = logits.shape[-1]

    end_index = jnp.zeros((n_beams,), dtype=jnp.int32)
    is_end = jnp.zeros((n_beams,), dtype=jnp.bool_)
    while i < max_len:
        input_ids = live_seqs[:, [i]]
        cache, logits = apply_fn(params, cache, input_ids)
        logits = logits.astype(jnp.float32)
        i += 1
        log_probs = jax.nn.log_softmax(logits[:, 0], axis=1)
        beam_log_probs = live_log_probs[:, None] + log_probs
        beam_log_probs = beam_log_probs.reshape((-1,))
        live_log_probs, topk_indices = jax.lax.top_k(beam_log_probs, n_beams)
        indices = topk_indices // vocab_size
        live_seqs = live_seqs[indices]
        cache = gather_beams(cache, indices)
        tokens = topk_indices % vocab_size
        live_seqs = live_seqs.at[:, i].set(tokens)

        is_end = is_end | (tokens == tokenizer.eos_token_id)
        end_index = jnp.where(
            is_end & (end_index == 0),
            jnp.full_like(end_index, i),
            end_index,
        )
        if jnp.all(is_end):
            break
    end_index = jnp.where(end_index == 0, max_len, end_index)
    live_seqs_t = jax.device_get(live_seqs)
    end_index = jax.device_get(end_index)
    return [tokenizer.decode(seq[:end_index[i]]) for i, seq in enumerate(live_seqs_t)]
