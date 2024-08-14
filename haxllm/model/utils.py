import math

from enum import Enum, auto
import jax
import jax.numpy as jnp

from haxllm.model.mixin import RoPEScalingConfig

class ModelTask(Enum):
    SequenceClassification = auto()
    LanguageModeling = auto()


def parse_task(task):
    if isinstance(task, ModelTask):
        return task
    assert isinstance(task, str), f"task must be a string, got {task}"
    task = task.lower()
    if task == "sequence_classification" or task == 'cls':
        return ModelTask.SequenceClassification
    elif task == "language_modeling" or task == 'lm':
        return ModelTask.LanguageModeling
    else:
        raise ValueError(f"Unknown task {task}")


def load_model_cls(module, task):
    task = parse_task(task)
    if task == ModelTask.SequenceClassification:
        return getattr(module, "TransformerSequenceClassifier")
    elif task == ModelTask.LanguageModeling:
        return getattr(module, "TransformerLMHeadModel")
    else:
        raise ValueError(f"Unknown task {task}")


def load_config(module, name, **kwargs):
    hub = getattr(module, "config_hub")
    config_cls = getattr(module, "TransformerConfig")
    if name in hub:
        config = hub[name]
    else:
        available = ", ".join(hub.keys())
        type_name = config_cls.__name__
        module_name = type_name.split(".")[-1]
        raise ValueError(f"Unknown {module_name} model {name}, available: {available}")
    return _load_config1(config_cls, config, **kwargs)


def _load_config1(cls, base_config, **kwargs):
    d = {**base_config}
    kwargs = {**kwargs}

    rope_scaling = d.get("rope_scaling", None)
    if rope_scaling is not None:
        assert isinstance(rope_scaling, dict)
        d["rope_scaling"] = RoPEScalingConfig(**rope_scaling)

    remat_scan = kwargs.get("remat_scan", False)
    remat = kwargs.get("remat", False)
    scan = kwargs.get("scan", False)
    assert not (remat_scan and (remat or scan)), \
        "Cannot use remat_scan with remat or scan"

    n_layers = d["n_layers"]
    lengths = kwargs.get("lengths", None)
    if remat_scan:
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
            assert math.prod(lengths) == n_layers, "sum of lengths must equal n_layers"
    elif scan:  # scan
        if lengths is None:
            lengths = (n_layers,)
        if isinstance(lengths, int):
            lengths = (lengths,)
        assert len(lengths) == 1, "lengths must be a tuple of length 1"
        if lengths[0] == -1:
            lengths = (n_layers,)
        else:
            assert lengths[0] == n_layers, "lengths must equal n_layers"
    else:
        lengths = []
    kwargs["lengths"] = tuple(lengths)

    ignored_keys = ["remat_policy"]
    for k, v in kwargs.items():
        if k in ignored_keys:
            continue
        if k not in cls.__dataclass_fields__:
            print(f"Unknown config key {k} for {cls.__name__}")
            continue
        d[k] = v
    
    if d['shard'] and "shard_cache" not in d:
        d['shard_cache'] = True
    return cls(**d)


def truncated_normal_init(stddev=1e-2, dtype=jnp.float_):
    from jax._src import core
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        stddev_ = jnp.asarray(stddev, dtype) / jnp.array(.87962566103423978, dtype)
        return jax.random.truncated_normal(key, -2, 2, named_shape, dtype) * stddev_
    return init


def calculate_num_params_from_pytree(params):
    params_sizes = jax.tree_util.tree_map(jnp.size, params)
    total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
    return total_parameters


def calculate_training_tflops(num_model_parameters, per_device_batch_size, max_len, hidden_size, n_layers):
    matmul_tflops = 6 * num_model_parameters * max_len * per_device_batch_size / 10**12
    attention_tflops = 12 * hidden_size * n_layers * max_len**2 * per_device_batch_size / 10**12
    total_tflops = matmul_tflops + attention_tflops
    print(f'Per train step, total TFLOPs will be {total_tflops:.2f}, split as {100 * matmul_tflops/total_tflops:.2f}% matmul',
        f'and {100 * attention_tflops/total_tflops:.2f}% attention')
    return matmul_tflops + attention_tflops


def infer_model_config(params):
    params = params['transformer']
    if 'hs' in params:
        # scan, length=1,2
        p = params['hs']['ln_1']['scale']
        hidden_size = p.shape[-1]
        n_layers = p.shape[0]
    elif 'hs_0' in params:
        # scan, length=3
        p = params['hs_0']['ln_1']['scale']
        hidden_size = p.shape[-1]
        n_layers = p.shape[0] * len([k for k in params.keys() if k.startswith('hs_')])
    else:
        p = params['h_0']['ln_1']['scale']
        hidden_size = p.shape[-1]
        n_layers = len([k for k in params.keys() if k.startswith('h_')])
    return hidden_size, n_layers


def report_params_and_flops(params, max_len, batch_size):
    n_devices = jax.local_device_count()
    per_device_batch_size = batch_size // n_devices
    num_model_parameters = calculate_num_params_from_pytree(params)
    print(f"number parameters: {num_model_parameters/10**9:.3f} billion")
    hidden_size, n_layers = infer_model_config(params)
    per_device_tflops = calculate_training_tflops(num_model_parameters, max_len, per_device_batch_size, hidden_size, n_layers)
    return per_device_tflops * n_devices
