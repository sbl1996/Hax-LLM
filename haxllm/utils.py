from typing import  Union
import math
import re
from datetime import datetime, timedelta
import collections
import itertools
from safetensors.numpy import load_file

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh

from flax import linen as nn
from flax import traverse_util
from flax.training import common_utils
from flax.core.frozen_dict import freeze, FrozenDict
from flax.traverse_util import unflatten_dict, flatten_dict

import optax


def spec_from_dataset(dataset, input_keys):
    if 'paddle' in str(type(dataset)):
        loader = dataset
        tensors = loader.dataset.tensor_dict
        batch_size = loader.batch_size
        return {
            key: {
                'shape': (batch_size,) + tensors[key].shape[1:],
                'dtype': tensors[key].dtype.name,
            } for key in input_keys
        }
    else:
        element_spec = dataset.element_spec
        return {
            key: {
                'shape': element_spec[key].shape.as_list(),
                'dtype': element_spec[key].dtype.name,
            } for key in input_keys
        }


def init_model(input_spec, model, init_rng):
    def init_fn(init_rng):
        inputs = {
            key: jnp.ones(spec['shape'], spec['dtype'])
            for key, spec in input_spec.items()
        }
        init_variables = model.init(init_rng, **inputs, train=False)
        return init_variables
    init_fn = jax.jit(init_fn)
    init_variables = init_fn(init_rng)
    return init_variables


def time_now():
    now = datetime.utcnow() + timedelta(hours=8)
    return now.strftime("%H:%M:%S")


def get_metrics(all_metrics, pmap=True):
    if pmap:
        all_metrics = common_utils.get_metrics(all_metrics)
        all_metrics = jax.tree_util.tree_map(np.sum, all_metrics)
    else:
        # metrics_np = jax.device_get(all_metrics)
        all_metrics = jax.tree_util.tree_map(
            lambda *xs: np.stack(xs).sum(), *all_metrics)
    denominator = all_metrics.pop('total')
    summary = jax.tree_util.tree_map(
        lambda x: x / denominator, all_metrics)
    return summary


def freeze_params_optimizer(optimizer, params, trainable_pattern):
    if not trainable_pattern:
        return optimizer
    optimizers = {'trainable': optimizer, 'frozen': optax.set_to_zero()}

    def match_label(path, v):
        path_str = ".".join(path)
        match = re.findall(trainable_pattern, path_str) != []
        if match:
            print(f"Trainable: {path_str}")
        return 'trainable' if match else 'frozen'

    param_labels = freeze(
        traverse_util.path_aware_map(match_label, params))
    tx = optax.multi_transform(optimizers, param_labels)
    return tx


def convert_scan_params(params, src_keys, tgt_keys, lengths):
    if isinstance(lengths, int):
        lengths = [lengths]
    scan_layers = math.prod(lengths)
    if scan_layers > len(src_keys):
        raise ValueError(
            f'Number of remat scan layers ({scan_layers}) must be less than or equal to the number of source keys ({len(src_keys)})')
    src_keys = src_keys[-scan_layers:]
    if len(lengths) == 1:
        level_trees = []
        for i in range(lengths[0]):
            level_trees.append(params.pop(src_keys[i]))
        params[tgt_keys] = jax.tree_map(
            lambda *xs: np.stack(xs, axis=0), *level_trees)
    elif len(lengths) == 2:
        level_trees = []
        for i in range(lengths[0]):
            loop_trees = []
            for j in range(lengths[1]):
                loop_trees.append(params.pop(src_keys[i * lengths[1] + j]))
            level_trees.append(jax.tree_map(
                lambda *xs: np.stack(xs, axis=0), *loop_trees))
        params[tgt_keys] = jax.tree_map(
            lambda *xs: np.stack(xs, axis=0), *level_trees)
    else:
        for l in range(lengths[0]):
            level_trees = []
            ol = l * lengths[1] * lengths[2]
            for i in range(lengths[1]):
                loop_trees = []
                for j in range(lengths[2]):
                    loop_trees.append(params.pop(src_keys[ol + i * lengths[2] + j]))
                level_trees.append(jax.tree_map(
                    lambda *xs: np.stack(xs, axis=0), *loop_trees))
            params[tgt_keys[l]] = jax.tree_map(
                lambda *xs: np.stack(xs, axis=0), *level_trees)
    return params


def pad(x, batch_size):
    b, *shape = x.shape
    rest = b % batch_size
    if rest:
        x = np.concatenate(
            [x, np.zeros((batch_size - rest, *shape), x.dtype)], axis=0)
    return x


def replace_val(p, x):
    if isinstance(p, nn.Partitioned):
        return p.replace_boxed(x)
    else:
        return x
 

def get_partition_spec(p: nn.Partitioned):
    return p.get_partition_spec()


def load_transformer_params(
        params,
        path: str,
        device,
        lm_head=False):
    cpu = False
    mesh = None
    if isinstance(device, str):
        assert device == 'cpu'
        cpu = True
    elif isinstance(device, Mesh):
        mesh = device

    transformer_params = load_file(path)
    transformer_params = unflatten_dict(transformer_params, sep=".")

    if isinstance(params, FrozenDict):
        params = params.unfreeze()
        frozen = True
    else:
        frozen = False
    params = flatten_dict(params, sep=".")

    # detect scan or remat_scan
    hs_keys = [k for k in params.keys() if k.startswith('transformer.hs')]
    if hs_keys:
        src_keys = [k for k in transformer_params if k.startswith('h_')]
        h_params = flatten_dict(transformer_params[src_keys[0]], sep=".")
        sample_key = next(iter(h_params.keys()))

        loop_hs_keys = [k for k in hs_keys if k.startswith("transformer.hs_")]
        if loop_hs_keys:
            n_loop = len(set([k.split('.')[1] for k in loop_hs_keys]))
            tgt_keys = [f"hs_{i}" for i in range(n_loop)]
            hs_param = params[f"transformer.hs_0.{sample_key}"]
            scan_lengths = (n_loop,)
        else:
            tgt_keys = 'hs'
            hs_param = params[f"transformer.hs.{sample_key}"]
            scan_lengths = ()
        if isinstance(hs_param, nn.Partitioned):
            hs_param = hs_param.value
        hs_shape = hs_param.shape
        h_shape = h_params[sample_key].shape
        del h_params, hs_param
        scan_lengths = scan_lengths + tuple(hs_shape[:(len(hs_shape) - len(h_shape))])
        src_keys = [f"h_{i}" for i in range(np.prod(scan_lengths))]
        transformer_params = convert_scan_params(
            transformer_params, src_keys=src_keys, tgt_keys=tgt_keys, lengths=scan_lengths)
    new_transformer_params = flatten_dict(transformer_params, sep=".")

    prefix = "transformer."
    all_keys = list([k[len(prefix):] for k in params.keys() if k.startswith(prefix)])
    if lm_head:
        all_keys.extend([ k for k in params.keys() if k.startswith("lm_head")])
    for key in all_keys:
        if key not in new_transformer_params:
            print(f"Key {key} not found in transformer params")
            continue
        
        full_key = key if key.startswith("lm_head") else prefix + key
        p = params[full_key]
        print(f"Loading param {key}")
        x = new_transformer_params[key]
        if isinstance(p, nn.Partitioned):
            dtype = p.value.dtype
        else:
            dtype = p.dtype
        x = x.astype(dtype)
        if cpu:
            x = jax.device_put(x, jax.devices("cpu")[0])
        elif mesh is None:
            x = jax.device_put(x)
        else:
            if isinstance(p, nn.Partitioned):
                spec = p.get_partition_spec()
            else:
                spec = p.sharding.spec
            #     print("Before p", p.value.sharding)
            #     print(p.names)
            # else:
            #     print("Before", p.sharding)
            with mesh:
                x = jax.device_put(x, NamedSharding(mesh, spec))
        if isinstance(p, nn.Partitioned):
            params[full_key] = p.replace_boxed(x)
        else:
            params[full_key] = x
        # if isinstance(p, nn.Partitioned):
        #     print("After p", params[full_key].value.sharding)
        #     print(params[full_key].names)
        # else:
        #     print("After", params[full_key].sharding)
        del p, x
    params = unflatten_dict(params, sep=".")
    if frozen:
        params = freeze(params)
    return params


def partition_x(x, mesh):
    n = len(x.shape)
    partition = ["X"] + [None] * (n - 1)
    return jax.device_put(x, NamedSharding(mesh, P(*partition)))


def prefetch_to_device(iterator, size, device):
    queue = collections.deque()

    def _prefetch(x):
        return jax.device_put(x, device)

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
        for data in itertools.islice(iterator, n):
            queue.append(jax.tree_util.tree_map(_prefetch, data))

    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)


def create_input_iter(ds, device, prepare_fn):
    it = map(prepare_fn, ds)
    it = prefetch_to_device(it, 2, device)
    return it
