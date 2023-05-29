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
    if key == 'remat_policy' and value not in ['default', 'minimal']:
        raise ValueError(f"Unknown remat policy {value}")
    _GCONFIG[key] = value


def get_remat_policy():
    remat_policy = get('remat_policy')
    if remat_policy == 'default':
        return None
    elif remat_policy == 'minimal':
        return jax.checkpoint_policies.dots_with_no_batch_dims_saveable