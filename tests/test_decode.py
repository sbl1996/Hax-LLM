import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')

from haxllm.model.decode import sample_token_top_p, sample_token_top_k, sample_token

def test_sample_token():
    rng = jax.random.PRNGKey(0)
    repeat = 10
    b, c = 8, 16
    print('test_sample_token')
    for i in range(repeat):
        rngs = jax.random.split(rng, b + 1)
        rng = rngs[0]
        subrngs = rngs[1:]
        logits = jax.random.normal(rng, (b, c))

        # sample_token_top_p
        p = 0.7
        tokens1 = sample_token_top_p(logits, subrngs, p)

        tokens2 = []
        for i in range(b):
            tokens = sample_token_top_p(logits[i], subrngs[i], p)
            tokens2.append(tokens)
        tokens2 = jnp.stack(tokens2)

        assert jnp.all(tokens1 == tokens2)

        # sample_token_top_k
        k = 5
        tokens1 = sample_token_top_k(logits, subrngs, k)

        tokens2 = []
        for i in range(b):
            tokens = sample_token_top_k(logits[i], subrngs[i], k)
            tokens2.append(tokens)
        tokens2 = jnp.stack(tokens2)

        assert jnp.all(tokens1 == tokens2)

        # sample_token
        tokens1 = sample_token(logits, subrngs, temperature=0.7)

        tokens2 = []
        for i in range(b):
            tokens = sample_token(logits[i], subrngs[i], temperature=0.7)
            tokens2.append(tokens)
        tokens2 = jnp.stack(tokens2)

        assert jnp.all(tokens1 == tokens2)

if __name__ == '__main__':
    test_sample_token()