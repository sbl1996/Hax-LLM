import jax
import jax.numpy as jnp

def load_config(cls, base_config, **kwargs):
    d = {**base_config}
    kwargs = {**kwargs}
    if 'scan_layers' in kwargs:
        if kwargs['scan_layers'] is None:
            kwargs['scan_layers'] = 0
        elif kwargs['scan_layers'] == -1:
            kwargs['scan_layers'] = d['n_layers']
    if 'remat_scan_lengths' in kwargs:
        remat_scan_lengths = kwargs['remat_scan_lengths']
        if remat_scan_lengths is not None:
            if len(remat_scan_lengths) == 1:
                remat_scan_lengths = (remat_scan_lengths[0], remat_scan_lengths[0])
    for k, v in kwargs.items():
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