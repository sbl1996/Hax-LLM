import optax

def get_optimizer(cfg, steps_per_epoch):
    init_lr = cfg.optimizer.warmup_min_lr
    peak_lr = cfg.optimizer.learning_rate
    warmup_steps = cfg.optimizer.warmup_steps
    if warmup_steps == 0:
        init_lr = peak_lr
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_lr, peak_lr, warmup_steps=warmup_steps, decay_steps=steps_per_epoch * cfg.epochs)

    optimizer = []
    if cfg.optimizer.clip_norm > 0:
        optimizer.append(optax.clip_by_global_norm(cfg.optimizer.clip_norm))
    opt = getattr(optax, cfg.optimizer.name)
    if cfg.optimizer.name in ['adamw', 'lion']:
        optimizer.append(opt(learning_rate=lr_schedule, weight_decay=cfg.optimizer.weight_decay, b1=cfg.optimizer.b1, b2=cfg.optimizer.b2))
    else:
        optimizer.append(opt(learning_rate=lr_schedule, weight_decay=cfg.optimizer.weight_decay))
    optimizer = optax.chain(*optimizer)
    return optimizer