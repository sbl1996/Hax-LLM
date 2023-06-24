import argparse

import importlib
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file
from flax.traverse_util import flatten_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-family", type=str, required=True,
                        help="Model family available in haxllm.model")
    parser.add_argument("-f", "--checkpoint-path", type=str, default=None,
                        help="Path to the model checkpoint file")
    parser.add_argument("-d", "--base-model-dir", type=str, default=None,
                        help="Path to model directory, usually a huggingface hub")
    parser.add_argument("-t", "--target-dir", type=str, default=".",
                        help="Directory to save to")

    args = parser.parse_args()
    mod_name = "haxllm.model.{}".format(args.model_family)
    print("Using {}".format(mod_name))
    mod = importlib.import_module(mod_name)
    target_dir = Path(args.target_dir).expanduser().absolute()

    if args.checkpoint_path is None and args.base_model_dir is None:
        raise ValueError("Must specify either --checkpoint-path or --base-model-dir")
    if args.checkpoint_path is not None and args.base_model_dir is not None:
        raise ValueError("Must specify either --checkpoint-path or --base-model-dir, not both")

    if args.checkpoint_path is not None:
        ckpt_path = Path(args.checkpoint_path).expanduser().absolute()
        suffix = ckpt_path.suffix
        assert suffix in [".bin", ".safetensors"], "Unknown file type: {}, must be .bin or .safetensors".format(suffix)
        bin_files = [ckpt_path]
        model_name = ckpt_path.stem
    else:
        model_dir = Path(args.base_model_dir).expanduser().absolute()
        bin_files = sorted(model_dir.glob("pytorch_model-*.bin"))
        if not bin_files:
            bin_files = sorted(model_dir.glob("model-*.safetensors"))
        if not bin_files:
            raise ValueError("No model files found in {}".format(model_dir))
        model_name = model_dir.name
    
    if bin_files[0].suffix == ".bin":
        file_type = "bin"
    else:
        file_type = "safetensors"

    tensors = {}
    if file_type == "safetensors":
        for f in bin_files:
            print("Loading from {}".format(f))
            with safe_open(f, framework="numpy", device='cpu') as f:
                for k in f.keys():
                    if k in tensors:
                        raise ValueError("Duplicate key: {}".format(k))
                    tensors[k] = f.get_tensor(k)
    else:
        import torch
        for f in bin_files:
            print("Loading from {}".format(f))
            d = torch.load(f, map_location="cpu")
            for k in list(d.keys()):
                if k in tensors:
                    raise ValueError("Duplicate key: {}".format(k))
                tensors[k] = d[k].numpy()
                del d[k]

    tensors = mod.remap_state_dict(tensors)
    tensors = flatten_dict(tensors, sep=".")
    save_path = target_dir / "{}_np.safetensors".format(model_name)
    print("Saving to {}".format(save_path))
    save_file(tensors, save_path)
