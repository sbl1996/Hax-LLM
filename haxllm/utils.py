import time
import math
import re
import struct
import json
from datetime import datetime, timedelta, timezone
import collections
import itertools
from tqdm import tqdm

from safetensors.numpy import load_file

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

from flax import linen as nn
from flax import traverse_util
from flax.training import common_utils
from flax.core.frozen_dict import freeze, FrozenDict
from flax.traverse_util import unflatten_dict, flatten_dict

import optax

from haxllm.model.utils import report_params_and_flops
from haxllm.model.quantize import unpack_q4


def spec_from_dataset(dataset, input_keys):
    if "paddle" in str(type(dataset)):
        loader = dataset
        tensors = loader.dataset.tensor_dict
        batch_size = loader.batch_size
        return {
            key: {
                "shape": (batch_size,) + tensors[key].shape[1:],
                "dtype": tensors[key].dtype.name,
            } for key in input_keys
        }
    else:
        element_spec = dataset.element_spec
        return {
            key: {
                "shape": element_spec[key].shape.as_list(),
                "dtype": element_spec[key].dtype.name,
            } for key in input_keys
        }


def time_now():
    now = datetime.now(timezone.utc) + timedelta(hours=8)
    return now.strftime("%H:%M:%S")


def get_metrics(all_metrics, pmap=True):
    if pmap:
        all_metrics = common_utils.get_metrics(all_metrics)
        all_metrics = jax.tree_util.tree_map(np.sum, all_metrics)
    else:
        # metrics_np = jax.device_get(all_metrics)
        all_metrics = jax.tree_util.tree_map(
            lambda *xs: np.stack(xs).sum(), *all_metrics)
    denominator = all_metrics.pop("total")
    summary = jax.tree_util.tree_map(
        lambda x: x / denominator, all_metrics)
    return summary


def freeze_params_optimizer(optimizer, params, trainable_pattern):
    if not trainable_pattern:
        return optimizer
    optimizers = {"trainable": optimizer, "frozen": optax.set_to_zero()}

    def match_label(path, v):
        path_str = ".".join(path)
        match = re.findall(trainable_pattern, path_str) != []
        if match:
            if isinstance(v, nn.Partitioned):
                print(f"Trainable: {path_str} {v.get_partition_spec()} {v.value.dtype}")
            else:
                print(f"Trainable: {path_str} {v.dtype}")
        return "trainable" if match else "frozen"

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
            f"Number of remat scan layers ({scan_layers}) must be less than or equal to the number of source keys ({len(src_keys)})")
    src_keys = src_keys[-scan_layers:]
    if len(lengths) == 1:
        level_trees = []
        for i in range(lengths[0]):
            level_trees.append(params.pop(src_keys[i]))
        params[tgt_keys] = jax.tree.map(
            lambda *xs: np.stack(xs, axis=0), *level_trees)
    elif len(lengths) == 2:
        level_trees = []
        for i in range(lengths[0]):
            loop_trees = []
            for j in range(lengths[1]):
                loop_trees.append(params.pop(src_keys[i * lengths[1] + j]))
            level_trees.append(jax.tree.map(
                lambda *xs: np.stack(xs, axis=0), *loop_trees))
        params[tgt_keys] = jax.tree.map(
            lambda *xs: np.stack(xs, axis=0), *level_trees)
    else:
        for l in range(lengths[0]):
            level_trees = []
            ol = l * lengths[1] * lengths[2]
            for i in range(lengths[1]):
                loop_trees = []
                for j in range(lengths[2]):
                    loop_trees.append(params.pop(src_keys[ol + i * lengths[2] + j]))
                level_trees.append(jax.tree.map(
                    lambda *xs: np.stack(xs, axis=0), *loop_trees))
            params[tgt_keys[l]] = jax.tree.map(
                lambda *xs: np.stack(xs, axis=0), *level_trees)
    return params


def pad(x, batch_size):
    b, *shape = x.shape
    rest = b % batch_size
    if rest:
        x = np.concatenate(
            [x, np.zeros((batch_size - rest, *shape), x.dtype)], axis=0)
    return x


def shard(xs):
    local_device_count = jax.local_device_count()
    return jax.tree_util.tree_map(
        lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), xs)


def inspect_tree(tree):
    flatten_tree = flatten_dict(tree, sep=".")
    for k, v in flatten_tree.items():
        if isinstance(v, nn.Partitioned):
            val = v.value
            print(f"{k}: {val.shape} {val.dtype}, {v.mesh}")
        else:
            print(f"{k}: {v.shape} {v.dtype}")


def get_sharding(mesh, fun, *args, **kwargs):
    xs = jax.eval_shape(fun, *args, **kwargs)
    return nn.get_sharding(xs, mesh)


def parse_safetensors_header(fp):
    with open(fp, 'rb') as f:
        # read the size of the header
        header_size_bytes = f.read(8)
        header_size = struct.unpack('Q', header_size_bytes)[0]

        # read the header
        header_bytes = f.read(header_size)
        header_str = header_bytes.decode('utf-8')
        header = json.loads(header_str)
    return header


def has_bf16_in_safetensors(fp):
    header = parse_safetensors_header(fp)
    return any([ d['dtype'] == 'BF16' for k, d in header.items() if 'dtype' in d])


def load_transformer_params(params, path: str, device, lm_head=False, verbose=False):
    fprint = print if verbose else lambda *args, **kwargs: None
    cpu = False
    mesh = None
    if isinstance(device, str):
        assert device == "cpu"
        cpu = True
    elif isinstance(device, Mesh):
        mesh = device

    if has_bf16_in_safetensors(path):
        np.bfloat16 = jnp.bfloat16
    transformer_params = load_file(path)
    transformer_params = unflatten_dict(transformer_params, sep=".")

    assert not isinstance(params, FrozenDict)

    # detect scan or remat_scan
    hs_keys = [k for k in params.keys() if k.startswith("transformer.hs")]
    if hs_keys:
        fprint("Converting scan params...")
        src_keys = [k for k in transformer_params if k.startswith("h_")]
        h_params = flatten_dict(transformer_params[src_keys[0]], sep=".")
        sample_key = next(iter(h_params.keys()))

        loop_hs_keys = [k for k in hs_keys if k.startswith("transformer.hs_")]
        if loop_hs_keys:
            n_loop = len(set([k.split(".")[1] for k in loop_hs_keys]))
            tgt_keys = [f"hs_{i}" for i in range(n_loop)]
            hs_param = params[f"transformer.hs_0.{sample_key}"]
            scan_lengths = (n_loop,)
        else:
            tgt_keys = "hs"
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
    all_keys.append("wte.embedding")
    if lm_head:
        all_keys.extend([ k for k in params.keys() if k.startswith("lm_head")])
    fprint("Loading param on device...")
    for key in tqdm(all_keys, disable=not verbose):
        if key not in new_transformer_params:
            fprint(f"Key {key} not found in transformer params")
            continue
        if key.startswith("lm_head") or key.startswith("wte"):
            full_key = key
        else:
            full_key = prefix + key
        p = params[full_key]
        x = new_transformer_params[key]
        # TODO: refactor with qconfig
        if x.dtype == jnp.int32 and "kernel" in key:
            x = unpack_q4(x, dtype=jnp.int8)
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
            with mesh:
                x = jax.device_put(x, nn.get_sharding(p, mesh))
        if isinstance(p, nn.Partitioned):
            params[full_key] = p.replace_boxed(x)
        else:
            params[full_key] = x
        del p, x
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


def create_mesh(mesh_shape, axis_names=("X", "Y")):
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices=device_mesh, axis_names=axis_names)
    return mesh


class MovingAverage:

    def __init__(self) -> None:
        self.total = 0
        self.count = 0
    
    def mark_start(self):
        self.start = time.time()
    
    def mark_end(self):
        self.total += time.time() - self.start
        self.count += 1
    
    def record(self, t):
        self.total += t
        self.count += 1
    
    def get_average(self):
        return self.total / self.count

    def reset(self):
        self.total = 0
        self.count = 0
