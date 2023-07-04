import os
import argparse

import importlib
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file

import jax.numpy as jnp
from flax.traverse_util import flatten_dict


def tensor_to_numpy(v):
    import torch
    if v.dtype == torch.bfloat16:
        v = v.float().numpy().astype(jnp.bfloat16)
    else:
        v = v.numpy()
    return v


def load_from_torch_bin_files(bin_files):
    tensors = {}
    import torch
    for f in bin_files:
        print("Loading from {}".format(f))
        d = torch.load(f, map_location="cpu")
        for k in list(d.keys()):
            if k in tensors:
                raise ValueError("Duplicate key: {}".format(k))
            tensors[k] = tensor_to_numpy(d[k])
            del d[k]
    return tensors


def load_from_safetensors_files(files):
    tensors = {}
    for f in files:
        print("Loading from {}".format(f))
        with safe_open(f, framework="numpy", device='cpu') as f:
            for k in f.keys():
                if k in tensors:
                    raise ValueError("Duplicate key: {}".format(k))
                tensors[k] = f.get_tensor(k)
    return tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-family", type=str, required=True,
                        help="Model family available in haxllm.model")
    parser.add_argument("-s", "--source", type=str, required=True,
                        help="Path to the model checkpoint file or directory, can be remote huggingface hub")
    parser.add_argument("-t", "--target", type=str, default=".",
                        help="Target path to save the dumped model")

    args = parser.parse_args()
    mod_name = "haxllm.model.{}".format(args.model_family)
    print("Using {}".format(mod_name))
    mod = importlib.import_module(mod_name)

    bin_files = None
    tensors = None

    if os.path.exists(os.path.expanduser(args.source)):
        source = Path(args.source).expanduser().absolute()
        if source.is_dir():
            model_dir = Path(args.base_model_dir).expanduser().absolute()
            bin_files = sorted(model_dir.glob("pytorch_model-*.bin"))
            if not bin_files:
                bin_files = sorted(model_dir.glob("model-*.safetensors"))
            if not bin_files:
                raise ValueError("No model files found in {}".format(model_dir))
            model_name = model_dir.name    
        else:
            ckpt_path = source
            suffix = ckpt_path.suffix
            assert suffix in [".bin", ".safetensors"], "Unknown file type: {}, must be .bin or .safetensors".format(suffix)
            bin_files = [ckpt_path]
            model_name = ckpt_path.stem
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.source, trust_remote_code=True)
        tensors = model.state_dict()
        del model
        keys = list(tensors.keys())
        for k in keys:
            tensors[k] = tensor_to_numpy(tensors[k])
        model_name = args.source.split("/")[-1]

    if tensors is None:
        if bin_files[0].suffix == ".bin":
            tensors = load_from_torch_bin_files(bin_files)
        else:
            tensors = load_from_safetensors_files(bin_files)

    target = Path(args.target).expanduser().absolute()
    if target.is_dir():
        save_path = target / "{}_np.safetensors".format(model_name)
    else:
        assert target.suffix == ".safetensors", "Unknown file type: {}, must be .safetensors".format(target.suffix)
        assert not target.exists(), "Target file already exists: {}".format(target)
        save_path = target

    tensors = mod.remap_state_dict(tensors)
    tensors = flatten_dict(tensors, sep=".")
    print("Saving to {}".format(save_path))
    save_file(tensors, save_path)
