from typing import Tuple
from flax import struct
import optax

from haxllm.optim import warmup_linear_decay_schedule

@struct.dataclass
class RematScanConfigMixin:
    remat: bool = False
    scan: bool = False
    remat_scan: bool = False
    lengths: Tuple[int, int] = (-1, 1)

    def remat_scan_lengths(self):
        if not self.remat_scan:
            raise ValueError("remat_scan_lengths called when remat_scan is False")
        return self.lengths

    def scan_lengths(self):
        if not self.scan:
            raise ValueError("scan_lengths called when scan is False")
        return self.lengths


def set_extra(extra, cfg, name):
    if hasattr(cfg, name):
        extra[name] = getattr(cfg, name)


def get_optimizer(cfg, steps_per_epoch):
    init_lr = cfg.optimizer.warmup_min_lr
    peak_lr = cfg.optimizer.learning_rate
    total_steps = steps_per_epoch * cfg.epochs
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

    optimizer.append(getattr(optax, opt_name)(learning_rate=lr_schedule, weight_decay=cfg.optimizer.weight_decay, **extras))
    optimizer = optax.chain(*optimizer)
    return optimizer