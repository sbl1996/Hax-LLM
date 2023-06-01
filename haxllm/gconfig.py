import jax

_GCONFIG = {
    'remat_scan_level': 2,
    'remat_policy': 'default',
}

def get(key):
    return _GCONFIG.get(key)

def set(key, value):
    if key not in _GCONFIG:
        raise KeyError(f"Unknown global config key {key}")
    if value is None:
        return
    if key == 'remat_policy':
        if value in ['default', 'minimal', 'none']:
            pass
        elif value.startswith("minimal-"):
            ratio = float(value[len("minimal-"):])
            if not 0 <= ratio <= 1:
                raise ValueError(f"Invalid remat ratio {ratio}")
        else:
            raise ValueError(f"Unknown remat policy {value}")
    _GCONFIG[key] = value


def save_partial_dots(state, ratio):
    from jax._src.lax import lax as lax_internal
    def policy(prim, *_, **params) -> bool:
        # This is a useful heuristic for transformers.
        if prim is lax_internal.dot_general_p:
            (_, _), (lhs_b, rhs_b) = params['dimension_numbers']
            if not lhs_b and not rhs_b:
                state['n'] += 1
                if state['c'] / state['n'] < ratio:
                    state['c'] += 1
                    return True
                else:
                    return False
        return False
    return policy


def get_remat_policy():
    remat_policy = get('remat_policy')
    if remat_policy == 'default':
        return None
    elif remat_policy == 'minimal':
        return jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    elif remat_policy.startswith("minimal-"):
        ratio = float(remat_policy[len("minimal-"):])
        return save_partial_dots({'n': 0, 'c': 0}, ratio)
    elif remat_policy == 'none':
        return jax.checkpoint_policies.everything_saveable