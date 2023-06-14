import jax
import jax.numpy as jnp

from flax.core.frozen_dict import unfreeze, FrozenDict


def add_beam_dim(x, n_beams):
    if x.ndim <= 2:
        return jnp.tile(x, [1] * x.ndim)
    reps = [1] * x.ndim
    reps[-4] = n_beams
    return jnp.tile(x, reps)


@jax.jit
def gather_beams(xs, indices):
    def gather_fn(x):
        if x.ndim <= 2:
            return x
        return x[..., indices, :, :, :]
    return jax.tree_map(gather_fn, xs)


def fix_cache_index(cache, true_index):
    if isinstance(cache, FrozenDict):
        cache = unfreeze(cache)
    v = cache['transformer']['hs']['attn']['cache_index']
    cache['transformer']['hs']['attn']['cache_index'] = jnp.full_like(v, true_index)
    return cache


def sample_token(logits, rng, temperature=1.0, topk=5):
    if temperature == 0:
        token = jnp.argmax(logits)
    elif topk > 1:
        logits, tokens = jax.lax.top_k(logits, topk)
        logits = logits / temperature
        rng, subrng = jax.random.split(rng)
        index = jax.random.categorical(subrng, logits, axis=-1)
        token = tokens[index]
    else:
        logits = logits / temperature
        rng, subrng = jax.random.split(rng)
        token = jax.random.categorical(subrng, logits, axis=-1)
    return token, rng


def random_sample(sentence, tokenizer, apply_fn, params, cache, max_len,
                  temperature=1.0, topk=10, rng=None, two_stage=False, pad_context=None, pad_token_id=None):
    if temperature != 0:
        assert rng is not None, "Must provide rng if temperature != 0"
    if pad_context is not None:
        assert two_stage, "Must use two_stage if pad_context is not None"
        assert pad_token_id is not None, "Must provide pad_token_id if pad_context is not None"
    example = tokenizer(sentence)
    tokens = example['input_ids']
    context_length = len(tokens)

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
        token, rng = sample_token(logits, rng, temperature, topk)
        live_seq = live_seq.at[:, context_length].set(token)
        if pad_context is not None:
            cache = fix_cache_index(cache, context_length)
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
        token, rng = sample_token(logits, rng, temperature, topk)
        live_seq = live_seq.at[:, i].set(token)
        if token in [tokenizer.eos_token_id, 0]:
            break
    live_seq = jax.device_get(live_seq[0, :i])
    return tokenizer.decode(live_seq)


def beam_search(sentence, tokenizer, apply_fn, params, cache, max_len,
                n_beams, two_stage=False, pad_context=None, pad_token_id=None):
    if pad_context is not None:
        assert two_stage, "Must use two_stage if pad_context is not None"
        assert pad_token_id is not None, "Must provide pad_token_id if pad_context is not None"
    example = tokenizer(sentence)
    tokens = example['input_ids']
    context_length = len(tokens)
    assert context_length < max_len, "Context length must be less than max_len"

    cache = jax.tree_map(lambda x: add_beam_dim(x, n_beams), cache)

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
            cache = fix_cache_index(cache, context_length)
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
