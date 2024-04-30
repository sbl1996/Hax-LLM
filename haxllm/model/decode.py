from functools import partial

import jax
import jax.numpy as jnp

from flax.core.frozen_dict import unfreeze, FrozenDict


def add_batch_dim(x, batch_size):
    r"""
    Add batch dimension to the input tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor.
        `x` maybe one of the following:
            - cached_key: (batch_size, num_heads, seq_len, head_dim)
            - cached_value: (batch_size, num_heads, seq_len, head_dim)
            - cached_index: ()
            - cache_position_ids: ()
            - cache_position: (batch_size,)
        With scan and remat_scan, 2 additional dimensions are added to the front.
    batch_size : int
        Batch size.
    """
    if x.ndim <= 3:
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
        keys = ['hs']
    else:
        keys = [ k for k in cache['transformer'] if k.startswith('h_') ]
    
    for key in keys:
        parent = cache['transformer'][key]['attn']
        parent['cache_index'] = parent['cache_index'] - offset
        if "cache_position" in parent:
            parent['cache_position'] = parent['cache_position'] - offset
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


@partial(jax.jit, static_argnums=(3,))
def sample_token_top_k_top_p_single(logits, rng, p, k):
    sorted_indices = jnp.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]

    topk_score = sorted_logits[-min(k, len(sorted_logits))]
    indices_to_remove = logits < topk_score
    big_neg = jnp.finfo(logits.dtype).min
    logits = jnp.where(indices_to_remove, big_neg, logits)

    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))
    mask = cumulative_probs > p
    mask = jnp.concatenate([jnp.zeros((1,), dtype=jnp.bool_), mask[:-1]])
    sorted_logits = jnp.where(mask, big_neg, sorted_logits)
    token = jax.random.categorical(rng, sorted_logits)
    token = sorted_indices[token]
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


@partial(jax.jit, static_argnums=(3,))
def sample_token_top_k_top_p(logits, rng, p, k):
    if logits.ndim == 1:
        return sample_token_top_k_top_p_single(logits, rng, p, k)
    elif logits.ndim == 2:
        if rng.ndim == 2:
            assert logits.shape[0] == rng.shape[0]
            return jax.vmap(sample_token_top_k_top_p_single, in_axes=(0, 0, None, None))(logits, rng, p, k)
        else:
            return jax.vmap(sample_token_top_k_top_p_single, in_axes=(0, None, None, None))(logits, rng, p, k)
    else:
        raise NotImplementedError


@jax.jit
def add_repeat_penalty(logits, input_ids, penalty=1.0):
    # Keep shape static to avoid recompilation
    ndim = logits.ndim
    if logits.ndim == 1:
        logits = logits[None, :]
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    assert logits.shape[0] == input_ids.shape[0] == 1, "Only support batch_size=1"
    logits = logits[0]
    input_ids = input_ids[0]
    new_logits = logits / jnp.where(
        logits > 0, penalty, -penalty)
    logits = logits.at[input_ids].set(new_logits[input_ids])
    if ndim == 2:
        logits = logits[None, :]
    return logits


def sample_token(logits, live_seq, rng, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1, repetition_penalty: float = 1.0):
    if temperature < 1e-5 or top_k == 1:
        return jnp.argmax(logits, axis=-1)
    if repetition_penalty != 1.0:
        logits = add_repeat_penalty(logits, live_seq, repetition_penalty)
    logits = logits / temperature
    if top_k > 1 and top_p < 1.0:
        return sample_token_top_k_top_p(logits, rng, top_p, top_k)
    elif top_k > 1:
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
                  temperature=1.0, top_k=5, top_p=1.0, repetition_penalty=1.0,
                  rng=None, two_stage=False, pad_context=None, pad_token_id=None,
                  decode=True, stop_token_ids=None, max_new_tokens=None):
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
    if tokens[0] == pad_token_id:
        raise ValueError("Padding left or empty input detected, not supported")
    context_length = len(tokens)
    assert context_length < max_len, "Context length must be less than max_len"

    cache = jax.tree_map(lambda x: jnp.tile(x, [1] * x.ndim), cache)

    live_seq = jnp.full((1, max_len), pad_token_id, dtype=jnp.int32)
    live_seq = live_seq.at[:, :context_length].set(tokens)

    i = 0
    new_tokens = 0
    while i < max_len:
        if two_stage and i == 0:
            first_len = context_length if pad_context is None else pad_context
            input_ids = live_seq[:, :first_len]
            cache, logits = apply_fn(params, cache, input_ids)
            if pad_context is not None:
                cache = fix_cache_index(cache, pad_context - context_length)
            logits = logits.astype(jnp.float32)[0, context_length-1]
            i = context_length
        else:
            input_ids = live_seq[:, [i]]
            cache, logits = apply_fn(params, cache, input_ids)
            logits = logits.astype(jnp.float32)[0, -1]
            i += 1
        if i < context_length:
            continue
        rng, subrng = split_rng(rng)
        token = sample_token(logits, live_seq, subrng, temperature, top_p, top_k, repetition_penalty)
        live_seq = live_seq.at[:, i].set(token)
        new_tokens += 1
        if max_new_tokens is not None and new_tokens >= max_new_tokens:
            break
        if token in stop_token_ids:
            break
    live_seq = jax.device_get(live_seq[0, :i])
    if decode:
        return tokenizer.decode(live_seq)
    else:
        return live_seq


def batch_random_sample(
    input_ids, apply_fn, params, cache, max_len, temperature=1.0, top_k=5, top_p=1.0,
    rng=None, pad_token_id=None, eos_token_id=None, stop_token_ids=None, padding_left=False,
    two_stage=None, pad_context=None, max_new_tokens=None):
    if stop_token_ids is None:
        stop_token_ids = [eos_token_id]
    if eos_token_id not in stop_token_ids:
        stop_token_ids.append(eos_token_id)

    if temperature < 1e-5:
        top_k = 1
    if temperature >= 1e-5 or top_k != 1:
        assert rng is not None, "Must provide rng if not using greedy decoding"

    assert pad_token_id is not None and eos_token_id is not None, "Must provide pad_token_id and eos_token_id" 
    
    if max_new_tokens is not None:
        assert padding_left is True, "Must use padding_left if max_new_tokens is not None"

    if padding_left:
        assert two_stage is not False, "Must use two_stage if padding_left is True"
        assert pad_context is None, "pad_context must be None if padding_left is True"
    else:
        assert two_stage is not True, "two_stage not supported if padding_left is False"

    batch_size, max_source_length = input_ids.shape

    context_length = jnp.argmax(input_ids == pad_token_id, axis=1)
    context_length = jnp.where(context_length == 0, max_source_length, context_length)

    cache = jax.tree_map(lambda x: add_batch_dim(x, batch_size), cache)

    live_seqs = jnp.full((batch_size, max_len), pad_token_id, dtype=jnp.int32)
    live_seqs = live_seqs.at[:, :max_source_length].set(input_ids)

    i = 0
    end_index = jnp.zeros((batch_size,), dtype=jnp.int32)
    is_end = jnp.zeros((batch_size,), dtype=jnp.bool_)
    new_tokens = 0
    while i < max_len:
        if padding_left and i == 0:
            input_ids = live_seqs[:, :max_source_length]
            cache, logits = apply_fn(params, cache, input_ids)
            i = max_source_length
        else:
            input_ids = live_seqs[:, [i]]
            cache, logits = apply_fn(params, cache, input_ids)
            i += 1
        logits = logits.astype(jnp.float32)[:, -1]
        if rng is None:
            subrngs = None
        else:
            rngs = jax.random.split(rng, batch_size + 1)
            rng, subrngs = rngs[0], rngs[1:]
        tokens = sample_token(logits, live_seqs, subrngs, temperature, top_p, top_k)
        is_context = i < context_length
        tokens = jnp.where(
            is_end | is_context, live_seqs[:, i], tokens)
        live_seqs = live_seqs.at[:, i].set(tokens)

        if padding_left and i >= max_source_length:
            new_tokens += 1
            if max_new_tokens is not None and new_tokens >= max_new_tokens:
                i += 1
                break

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
