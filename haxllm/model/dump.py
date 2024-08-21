import os
import json
from typing import List, Optional
from dataclasses import dataclass
import tyro

import importlib
from pathlib import Path

import numpy as np

from safetensors import safe_open
from safetensors.numpy import save_file

import jax.numpy as jnp
from flax.traverse_util import flatten_dict

from haxllm.utils import has_bf16_in_safetensors
from haxllm.model.quantize import QConfig


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
        print("Load from {}".format(f))
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
        print("Load from {}".format(f))
        if has_bf16_in_safetensors(f):
            np.bfloat16 = jnp.bfloat16
        with safe_open(f, framework="numpy", device='cpu') as f:
            for k in f.keys():
                if k in tensors:
                    raise ValueError("Duplicate key: {}".format(k))
                tensors[k] = f.get_tensor(k)
    return tensors

SAFE_TENSORS_INDEX = "model.safetensors.index.json"
TORCH_INDEX = "pytorch_model.bin.index.json"
SINGLE_SAFE_TENSORS = "model.safetensors"
SINGLE_TORCH = "pytorch_model.bin"

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
        single_file = model_dir / SINGLE_TORCH
    elif ckpt_type == "safetensors":
        index_file = model_dir / SAFE_TENSORS_INDEX
        single_file = model_dir / SINGLE_SAFE_TENSORS
    else:
        raise ValueError("Unknown ckpt type: {}".format(ckpt_type))
    if index_file.exists():
        print(ckpt_type, index_file, single_file)
        ckpt_files = get_ckpt_files_from_index(index_file)
        return all((model_dir / f).exists() for f in ckpt_files)
    elif single_file.exists():
        return True
    return False


def check_safetensors_ckpt(model_dir):
    return check_ckpt(model_dir, "safetensors")


def check_torch_ckpt(model_dir):
    return check_ckpt(model_dir, "bin")

@dataclass
class Args:
    source: str
    """Path to the model checkpoint file or directory, can be remote huggingface hub"""
    output: str = "."
    """Output path to save the dumped model"""
    format: str = "llama"
    """weight format, typically llama"""
    type: Optional[str] = None
    """Checkpoint type, can be either None, bin or safetensors. If None, will try to infer from the file extension"""
    dim: Optional[int] = None
    """Dimension of the head, if not specified, will try to infer from the model config"""
    qconfig: Optional[str] = None
    """Path to the qconfig file, if not specified, do not quantize the model"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    bin_files = None
    model_name = None

    if args.qconfig:
        with open(args.qconfig, 'r') as f:
            qconfig = QConfig.from_json(f.read())
    else:
        qconfig = None

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
                    try:
                        model = AutoModel.from_pretrained(source, trust_remote_code=True)
                    except ImportError as e:
                        raise ValueError("Cannot load model from huggingface hub, try to specify --type, original error: {}".format(e))
                del model
        else:
            ckpt_exists = check_ckpt(model_dir, args.type)
            if not ckpt_exists:
                from huggingface_hub import hf_hub_download
                filename = SAFE_TENSORS_INDEX if args.type == "safetensors" else TORCH_INDEX
                ckpt_index = hf_hub_download(repo_id=source, filename=filename)
                ckpt_files = get_ckpt_files_from_index(ckpt_index)
                for f in ckpt_files:
                    hf_hub_download(repo_id=source, filename=f)

        source = os.path.dirname(config_file)
        model_name = args.source.split("/")[-1]

    source = Path(source).expanduser().absolute()
    if source.is_dir():
        model_dir = source
        bin_files = sorted(model_dir.glob("pytorch_model*.bin"))
        if not bin_files:
            bin_files = sorted(model_dir.glob("model*.safetensors"))
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

    if qconfig is not None:
        print("Quantize model with {}".format(qconfig))
    from haxllm.model.llama import remap_state_dict
    tensors = remap_state_dict(tensors, head_dim=args.dim, qconfig=qconfig, format=args.format)
    tensors = flatten_dict(tensors, sep=".")
    print("Save model to {}".format(save_path))
    save_file(tensors, save_path)