import optax


def set_extra(extra, cfg, name):
    if hasattr(cfg, name):
        extra[name] = getattr(cfg, name)


def get_optimizer(cfg, steps_per_epoch):
    init_lr = cfg.optimizer.warmup_min_lr
    peak_lr = cfg.optimizer.learning_rate
    warmup_steps = cfg.optimizer.warmup_steps
    print(f"init_lr: {init_lr}, peak_lr: {peak_lr}, warmup_steps: {warmup_steps}")
    if warmup_steps == 0:
        init_lr = peak_lr
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_lr, peak_lr, warmup_steps=warmup_steps, decay_steps=steps_per_epoch * cfg.epochs)

    optimizer = []
    if cfg.optimizer.clip_norm > 0:
        optimizer.append(optax.clip_by_global_norm(cfg.optimizer.clip_norm))
        
    opt_name = cfg.optimizer.name

    extras = {}
    if opt_name in ['adamw', 'lion']:
        set_extra(extras, cfg.optimizer, 'b1')
        set_extra(extras, cfg.optimizer, 'b2')
        set_extra(extras, cfg.optimizer, 'mu_dtype')
    if opt_name == 'adamw':
        set_extra(extras, cfg.optimizer, 'eps')

    optimizer.append(getattr(optax, opt_name)(learning_rate=lr_schedule, weight_decay=cfg.optimizer.weight_decay, **extras))
    optimizer = optax.chain(*optimizer)
    return optimizer