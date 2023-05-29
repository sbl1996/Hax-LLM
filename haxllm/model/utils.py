import math

import jax
import jax.numpy as jnp

def load_config(cls, base_config, **kwargs):
    d = {**base_config}
    kwargs = {**kwargs}
    assert not (kwargs['remat_scan'] and (kwargs['remat'] or kwargs['scan'])), \
        "Cannot use remat_scan with remat or scan"

    n_layers = d['n_layers']
    lengths = kwargs.get('lengths', None)
    if kwargs['remat_scan']:
        if lengths is None:
            lengths = (n_layers, 1)
        assert len(lengths) in [2, 3], "remat_scan_lengths must be a tuple of length 2 or 3" 
        # replace -1 with the actual length
        if -1 in lengths:
            lengths = list(lengths)
            n = n_layers / -math.prod(lengths)
            assert n == int(n), "n_layers must be divisible by the sum of lengths"
            lengths[lengths.index(-1)] = int(n)
            lengths = tuple(lengths)
        else:
            assert sum(lengths) == n_layers, "sum of lengths must equal n_layers"
    else:  # scan
        if lengths is None:
            lengths = (n_layers,)
        if isinstance(lengths, int):
            lengths = (lengths,)
        assert len(lengths) == 1, "lengths must be a tuple of length 1"
        if lengths[0] == -1:
            lengths = (n_layers,)
        else:
            assert lengths[0] == n_layers, "lengths must equal n_layers"
    print(f"Using lengths {lengths}")
    kwargs['lengths'] = lengths

    for k, v in kwargs.items():
        if k not in cls.__dataclass_fields__:
            print(f"Unknown config key {k} for {cls.__name__}")
            continue
        d[k] = v
    return cls(**d)


def truncated_normal_init(stddev=1e-2, dtype=jnp.float_):
    from jax._src import core
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        stddev_ = jnp.asarray(stddev, dtype) / jnp.array(.87962566103423978, dtype)
        return jax.random.truncated_normal(key, -2, 2, named_shape, dtype) * stddev_
    return init