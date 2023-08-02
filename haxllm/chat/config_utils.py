import os
import importlib

from omegaconf import DictConfig, OmegaConf

import jax
import jax.numpy as jnp

from transformers import AutoTokenizer

from haxllm.pipeline.text_generation import ChatPipeline

def load_config(cfg: DictConfig):
    model_name = getattr(cfg, "model", None)
    if model_name is None:
        raise ValueError("Model name not specified")

    template = cfg.template
    # TODO: infer template from model name
    # template = getattr(cfg, "template", None)
    # if template is None:
    #     print("Template not specified, infer from model name...")
    #     from hydra.core.hydra_config import HydraConfig
    #     hydra_cfg = HydraConfig.get()
    #     main_path = [ s['path'] for s in hydra_cfg.runtime['config_sources'] if s['provider'] == 'main' ][0]
    #     all_templates = [
    #         os.path.basename(p).replace(".yaml", "")
    #         for p in glob.glob(os.path.join(main_path, "template", "*.yaml"))
    #     ]
    #     template_names = difflib.get_close_matches(model_name, all_templates, n=1, cutoff=0.5)
    #     if len(template_names) == 0:
    #         raise ValueError(f"Cannot infer template from model name {model_name}")
    #     template = template_names[0]

    template_config = OmegaConf.to_container(template, resolve=True)
    random_seed = cfg.seed
    tokenizer_name = template_config.pop("tokenizer")
    conv_template = template_config.pop("conv_template", None)

    checkpoint = getattr(cfg, "checkpoint", None)
    if checkpoint is None:
        checkpoint = model_name + "_np.safetensors"
        print(f"Checkpoint not specified, follow model config, using {checkpoint}")
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    temperature = getattr(cfg, "temperature", 1.0)
    top_p = getattr(cfg, "top_p", 1.0)
    top_k = getattr(cfg, "top_k", -1)

    platform = jax.default_backend()

    mesh = getattr(cfg, "mesh", None)
    if mesh == 'auto':
        mesh = None if platform == 'cpu' else [1, jax.local_device_count()]

    dtype = cfg.dtype
    if dtype == 'auto':
        dtype = jnp.bfloat16 if platform == 'tpu' else jnp.float16

    param_dtype = cfg.param_dtype
    if param_dtype == 'auto':
        param_dtype = jnp.bfloat16 if platform == 'tpu' else jnp.float16

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.decode(tokenizer("init")['input_ids'])

    parallel = mesh is not None

    module = "haxllm.model"
    peft = getattr(cfg, "peft", None)
    if peft is not None:
        module = module + "." + peft
    mod = importlib.import_module(module + "." + template_config.pop("family"))

    model_config = {"name": model_name, **template_config}
    config = getattr(mod, "load_config")(
        dtype=jnp.dtype(dtype),
        param_dtype=jnp.dtype(param_dtype),
        **model_config,
        decode=True,
        shard=parallel,
        shard_cache=parallel,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.pad_token_id

    model = getattr(mod, "TransformerLMHeadModel")(config)

    max_len = getattr(cfg, "max_len", None) or config.n_positions
    pipeline = ChatPipeline(
        tokenizer, model, max_len=max_len, seed=random_seed,
        temperature=temperature, top_p=top_p, top_k=top_k)
    pipeline.init(transformer_weight=checkpoint, mesh=mesh)
    return pipeline, conv_template