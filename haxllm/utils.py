import math
import re
from datetime import datetime, timedelta

import numpy as np

import jax
import jax.numpy as jnp

from flax import traverse_util
from flax.training import common_utils
from flax.core.frozen_dict import freeze
from flax.traverse_util import unflatten_dict, flatten_dict

import optax


def spec_from_dataset(dataset, input_keys):
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
    now = datetime.now() + timedelta(hours=8)
    return now.strftime("%H:%M:%S")


def get_metrics(all_metrics, pmap=True):
    if pmap:
        all_metrics = common_utils.get_metrics(all_metrics)
        all_metrics = jax.tree_util.tree_map(jnp.sum, all_metrics)
    else:
        metrics_np = jax.device_get(all_metrics)
        all_metrics = jax.tree_util.tree_map(
            lambda *xs: np.stack(xs).sum(), *metrics_np)
    denominator = all_metrics.pop('total')
    summary = jax.tree_util.tree_map(
        lambda x: x / denominator, all_metrics)
    return summary


def freeze_params_optimizer(optimizer, params, trainable_pattern):
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


def convert_remat_scan_params(params, src_keys, tgt_key, lengths):
    assert len(
        lengths) == 2, 'Only 2D remat scan is supported, because 1D remat_scan is equivalent to scan, WITHOUT remat.'
    remat_scan_layers = math.prod(lengths)
    if remat_scan_layers > len(src_keys):
        raise ValueError(
            f'Number of remat scan layers ({remat_scan_layers}) must be less than or equal to the number of source keys ({len(src_keys)})')
    src_keys = src_keys[-remat_scan_layers:]
    level_trees = []
    for i in range(lengths[0]):
        loop_trees = []
        for j in range(lengths[1]):
            loop_trees.append(params.pop(src_keys[i * lengths[1] + j]))
        level_trees.append(jax.tree_map(
            lambda *xs: np.stack(xs, axis=0), *loop_trees))
    params[tgt_key] = jax.tree_map(
        lambda *xs: np.stack(xs, axis=0), *level_trees)
    return params



def convert_scan_params(params, src_keys, tgt_key, lengths):
    if isinstance(lengths, int):
        lengths = [lengths]
    scan_layers = math.prod(lengths)
    if scan_layers > len(src_keys):
        raise ValueError(
            f'Number of remat scan layers ({scan_layers}) must be less than or equal to the number of source keys ({len(src_keys)})')
    src_keys = src_keys[-scan_layers:]
    level_trees = []
    if len(lengths) == 1:
        for i in range(lengths[0]):
            level_trees.append(params.pop(src_keys[i]))
    else:
        for i in range(lengths[0]):
            loop_trees = []
            for j in range(lengths[1]):
                loop_trees.append(params.pop(src_keys[i * lengths[1] + j]))
            level_trees.append(jax.tree_map(
                lambda *xs: np.stack(xs, axis=0), *loop_trees))
    params[tgt_key] = jax.tree_map(
        lambda *xs: np.stack(xs, axis=0), *level_trees)
    return params


def pad(x, batch_size):
    b, *shape = x.shape
    rest = b % batch_size
    if rest:
        x = np.concatenate(
            [x, np.zeros((batch_size - rest, *shape), x.dtype)], axis=0)
    return x


# def pad_partition_unpad(wrapped, partition_spec, static_argnums=(0,), static_argnames=()):
#     def pad_shard_unpad_wrapper(*args, batch_size=None, **kw):
#         batch_sizes = set()
#         for i, a in enumerate(args):
#             if i not in static_argnums:
#                 batch_sizes |= {t.shape[0] for t in jax.tree_util.tree_leaves(a)}
#         for k, v in kw.items():
#             if k not in static_argnames:
#                 batch_sizes |= {t.shape[0] for t in jax.tree_util.tree_leaves(v)}
#         assert len(batch_sizes) == 1, f"Inconsistent batch-sizes: {batch_sizes}"
#         b = batch_sizes.pop()

#         def pad(x):
#             _, *shape = x.shape
#             rest = batch_size - b
#             if rest:
#                 x = np.concatenate(
#                     [x, np.zeros((rest, *shape), x.dtype)], axis=0)
#             return nn_partitioning.with_sharding_constraint(x, partition_spec)

#         def maybe_pad(tree, actually_pad=True):
#             if not actually_pad: return tree  # For call-site convenience below.
#             return jax.tree_util.tree_map(pad, tree)

#         args = [maybe_pad(a, i not in static_argnums) for i, a in enumerate(args)]
#         kw = {k: maybe_pad(v, k not in static_argnames) for k, v in kw.items()}
#         out = wrapped(*args, **kw)
#         print(out)

#         def unpad(x):
#             return jax.device_get(x)[:b]
#         return jax.tree_util.tree_map(unpad, out)
#     return pad_shard_unpad_wrapper


def merge_pretrained_transformer_params(params, transformer_params, n_layer, scan_lengths):
    transformer_params = unflatten_dict(transformer_params, sep='.')
    if scan_lengths:
        transformer_params = convert_scan_params(
            transformer_params, src_keys=[f'h_{i}' for i in range(n_layer)],
            tgt_key='hs', lengths=scan_lengths)

    params = params.unfreeze()
    old_transformer_params = flatten_dict(params['transformer'], sep=".")
    new_transformer_params = flatten_dict(transformer_params, sep=".")
    all_keys = list(old_transformer_params.keys())
    for key in all_keys:
        if key in new_transformer_params:
            x = new_transformer_params[key]
            p = old_transformer_params[key]
            dtype = p.dtype
            sharding = p.sharding
            del old_transformer_params[key], p
            p = jax.device_put(jnp.asarray(x, dtype=dtype), sharding)
            old_transformer_params[key] = p

    transformer_params = unflatten_dict(old_transformer_params, sep=".")
    params['transformer'] = transformer_params
    params = freeze(params)
    return params