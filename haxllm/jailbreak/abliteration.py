from tqdm import tqdm

import jax
from flax.traverse_util import flatten_dict, unflatten_dict

import numpy as np


@jax.jit
def get_orthogonalized_matrix(matrix, vec):
    k = matrix @ vec[:, None]
    return matrix - k * vec, k


def reconstruct_params(pipeline, refusal_dirs, cache):
    params = pipeline.params
    pipeline.params = None
    params = flatten_dict(params, sep=".")

    keys = list(params.keys())
    for k in keys:
        if any(t in k for t in ['wte', 'attn.out', 'mlp.down']):
            v = params[k]
            w = v.value + cache[k] * refusal_dirs
            params[k] = v.replace_boxed(w)

    params = unflatten_dict(params, sep=".")
    pipeline.params = params


def add_refusal_directions(pipeline, refusal_dirs):
    params = pipeline.params
    pipeline.params = None
    params = flatten_dict(params, sep=".")
    cache = {}

    keys = list(params.keys())
    for k in keys:
        if any(t in k for t in ['wte', 'attn.out', 'mlp.down']):
            v = params[k]
            w, t = get_orthogonalized_matrix(v.value, refusal_dirs)
            params[k] = v.replace_boxed(w)
            cache[k] = t

    params = unflatten_dict(params, sep=".")
    pipeline.params = params
    return cache


def get_act_idx(cache_dict, act_name, layer):
    return cache_dict[f"transformer.h_{layer}.{act_name}"]


def get_refusal_directions(pipeline, harmful_tokens, harmless_tokens, batch_size=32, activation_layers=['resid_pre'], progress_bar=False):
    harmful = {}
    harmless = {}
    n = len(harmful_tokens)
    it = range(0, n // batch_size + (n % batch_size > 0))
    if progress_bar:
        it = tqdm(it)
    for i in it:
        start = i * batch_size
        end = min(n, start + batch_size)

        harmful_features = pipeline.forward(harmful_tokens[start:end], inspect=True)[-1]
        harmless_features = pipeline.forward(harmless_tokens[start:end], inspect=True)[-1]
        
        harmful_features = flatten_dict(harmful_features, sep=".")
        harmless_features = flatten_dict(harmless_features, sep=".")
        for key in harmful_features:
            harmful_v = np.array(harmful_features[key][0])
            harmless_v = np.array(harmless_features[key][0])
            if key not in harmful:
                harmful[key] = [harmful_v]
                harmless[key] = [harmless_v]
            else:
                harmful[key].append(harmful_v)
                harmless[key].append(harmless_v)

    harmful = {k: np.concatenate(v) for k,v in harmful.items()}
    harmless = {k: np.concatenate(v) for k,v in harmless.items()}

    refusal_dirs = { k: [] for k in activation_layers }

    for layer_num in range(pipeline.model.config.n_layers):
        pos = -1

        for layer in activation_layers:
            harmful_mean_act = get_act_idx(harmful, layer, layer_num)[:, pos, :].mean(axis=0)
            harmless_mean_act = get_act_idx(harmless, layer, layer_num)[:, pos, :].mean(axis=0)
        
            refusal_dir = harmful_mean_act - harmless_mean_act
            refusal_dir = refusal_dir / np.linalg.norm(refusal_dir)
            refusal_dirs[layer].append(refusal_dir)
    return refusal_dirs
