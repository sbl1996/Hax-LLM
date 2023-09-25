from omegaconf import OmegaConf

import optax

from haxllm.optim import warmup_linear_decay_schedule
from haxllm.config_utils import get_module
from haxllm.chat.setting import get_chat_setting


def set_extra(extra, cfg, name):
    if hasattr(cfg, name):
        extra[name] = getattr(cfg, name)


def load_optimizer(cfg, steps_per_epoch):
    init_lr = cfg.optimizer.warmup_min_lr
    peak_lr = cfg.optimizer.learning_rate
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = getattr(cfg.optimizer, "warmup_steps", 0)
    if warmup_steps == 0 and hasattr(cfg.optimizer, "warmup_ratio"):
        warmup_steps = int(total_steps * cfg.optimizer.warmup_ratio)
    min_ratio = getattr(cfg.optimizer, "min_ratio", 0.0)
    print(f"init_lr: {init_lr}, peak_lr: {peak_lr}, warmup_steps: {warmup_steps}")
    if warmup_steps == 0:
        init_lr = peak_lr
    schedule = getattr(cfg.optimizer, "schedule", "cosine")
    if schedule == "cosine":
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_lr, peak_lr, warmup_steps=warmup_steps, decay_steps=total_steps,
            end_value=peak_lr * min_ratio)
    elif schedule == "linear":
        lr_schedule = warmup_linear_decay_schedule(
            init_lr, peak_lr, warmup_steps=warmup_steps, decay_steps=total_steps,
            end_value=peak_lr * min_ratio)
    else:
        raise ValueError(f"Unknown schedule {schedule}")

    optimizer = []
    if cfg.optimizer.clip_norm > 0:
        optimizer.append(optax.clip_by_global_norm(cfg.optimizer.clip_norm))
        
    opt_name = cfg.optimizer.name

    extras = {}
    if opt_name in ["adamw", "lion"]:
        set_extra(extras, cfg.optimizer, "b1")
        set_extra(extras, cfg.optimizer, "b2")
        set_extra(extras, cfg.optimizer, "mu_dtype")
    if opt_name == "adamw":
        set_extra(extras, cfg.optimizer, "eps")

    if opt_name == 'sgdw':
        from haxllm.optim import sgdw
        optimizer.append(sgdw(
            learning_rate=lr_schedule, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay,
            nesterov=cfg.optimizer.nesterov, accumulator_dtype=cfg.optimizer.accumulator_dtype))
    else:
        optimizer.append(getattr(optax, opt_name)(learning_rate=lr_schedule, weight_decay=cfg.optimizer.weight_decay, **extras))
    optimizer = optax.chain(*optimizer)
    optimizer.trainable_pattern = getattr(cfg.train, "trainable_pattern", None)
    return optimizer


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    family = cfg.family
    peft = None
    if hasattr(cfg, "peft"):
        peft = cfg.peft.method
    mod = get_module(family, peft)
    model_pad_token_id = getattr(getattr(mod, "TransformerConfig"), "pad_token_id", None)

    tokenizer_config = cfg.tokenizer
    tokenizer_name = tokenizer_config.name

    print(f"Loading tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.padding_side = getattr(tokenizer_config, "padding_side", "right")
    tokenizer.truncation_side = getattr(tokenizer_config, "truncation_side", "right")
    tokenizer.decode(tokenizer("init")['input_ids'])

    if model_pad_token_id is not None and tokenizer.pad_token_id != model_pad_token_id:
        print(f"Changing tokenizer pad token id from {tokenizer.pad_token_id} to {model_pad_token_id}")
        tokenizer.pad_token_id = model_pad_token_id

    return tokenizer


def load_prompt_template(cfg_prompt_template):
    if cfg_prompt_template is not None:
        with open(cfg_prompt_template, "r") as f:
            prompt_template = f.read()
    else:
        prompt_template = None
    return prompt_template


def load_chat(cfg, eos_token_id):
    if hasattr(cfg, "chat"):
        chat_config: dict = OmegaConf.to_container(cfg.chat, resolve=True)
        chat_setting = get_chat_setting(chat_config.pop("conv_template"))
        stop_token_ids = list(chat_setting.stop_token_ids)
        if eos_token_id not in stop_token_ids:
            stop_token_ids.append(eos_token_id)
        prompt_template = load_prompt_template(chat_config.pop("prompt_template", None))
        chat_args = {
            "stop_token_ids": chat_setting.stop_token_ids, **chat_config}
    else:
        chat_setting = None
        chat_args = {
            "stop_token_ids": [eos_token_id],
            "temperature": 0.0,
        }
        prompt_template = None
    return chat_args, chat_setting, prompt_template


def load_peft(cfg):
    if hasattr(cfg, "peft"):
        peft_config: dict = OmegaConf.to_container(cfg.peft, resolve=True)
        name = peft_config.pop("method")
        kwargs = {**peft_config}
    else:
        name = None
        kwargs = {}
    return name, kwargs