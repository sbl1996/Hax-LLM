import os
import json
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

SAFE_TENSORS_INDEX = "model.safetensors.index.json"
TORCH_INDEX = "pytorch_model.bin.index.json"


def get_ckpt_files_from_index(index_file):
    with open(index_file) as f:
        d = json.load(f)
        if "weight_map" in d:
            d = d["weight_map"]
        ckpt_files = list(set(d.values()))
    return ckpt_files


def check_ckpt(model_dir, ckpt_type):
    if ckpt_type == "bin":
        index_file = model_dir / TORCH_INDEX
    elif ckpt_type == "safetensors":
        index_file = model_dir / SAFE_TENSORS_INDEX
    else:
        raise ValueError("Unknown ckpt type: {}".format(ckpt_type))
    if not index_file.exists():
        return False
    ckpt_files = get_ckpt_files_from_index(index_file)
    return all((model_dir / f).exists() for f in ckpt_files)


def check_safetensors_ckpt(model_dir):
    return check_ckpt(model_dir, "safetensors")


def check_torch_ckpt(model_dir):
    return check_ckpt(model_dir, "bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-family", type=str, required=True,
                        help="Model family available in haxllm.model")
    parser.add_argument("-s", "--source", type=str, required=True,
                        help="Path to the model checkpoint file or directory, can be remote huggingface hub")
    parser.add_argument("-o", "--output", type=str, default=".",
                        help="Output path to save the dumped model")
    parser.add_argument("-t", "--type", type=str, default=None,
                        help="Checkpoint type, can be either None, bin or safetensors. If None, will try to infer from the file extension")


    args = parser.parse_args()
    mod_name = "haxllm.model.{}".format(args.model_family)
    print("Using {}".format(mod_name))
    mod = importlib.import_module(mod_name)

    bin_files = None
    model_name = None

    source = args.source
    if not os.path.exists(os.path.expanduser(source)):
        print("Not a local path, try to load from huggingface hub")
        from transformers import AutoModelForCausalLM, AutoConfig, AutoModel
        from transformers.utils.hub import cached_file
        config = AutoConfig.from_pretrained(source, trust_remote_code=True)
        config_file = cached_file(source, "config.json")
        model_dir = Path(config_file).parent

        if args.type is None:
            ckpt_exists = check_torch_ckpt(model_dir) or check_safetensors_ckpt(model_dir)
            if not ckpt_exists:
                try:
                    model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True)
                except ValueError:
                    model = AutoModel.from_pretrained(source, trust_remote_code=True)
                del model
        else:
            ckpt_exists = check_ckpt(model_dir, args.type)
            if not ckpt_exists:
                from huggingface_hub import hf_hub_download
                ckpt_index = hf_hub_download(repo_id=source, filename=SAFE_TENSORS_INDEX)
                ckpt_files = get_ckpt_files_from_index(ckpt_index)
                for f in ckpt_files:
                    hf_hub_download(repo_id=source, filename=f)

        source = os.path.dirname(config_file)
        model_name = args.source.split("/")[-1]

    source = Path(source).expanduser().absolute()
    if source.is_dir():
        model_dir = source
        bin_files = sorted(model_dir.glob("pytorch_model-*.bin"))
        if not bin_files:
            bin_files = sorted(model_dir.glob("model-*.safetensors"))
        if not bin_files:
            raise ValueError("No model files found in {}".format(model_dir))
        model_name = model_name or model_dir.name    
    else:
        ckpt_path = source
        suffix = ckpt_path.suffix
        assert suffix in [".bin", ".safetensors"], "Unknown file type: {}, must be .bin or .safetensors".format(suffix)
        bin_files = [ckpt_path]
        model_name = model_name or ckpt_path.stem

    if bin_files[0].suffix == ".bin":
        tensors = load_from_torch_bin_files(bin_files)
    else:
        tensors = load_from_safetensors_files(bin_files)

    target = Path(args.output).expanduser().absolute()
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
