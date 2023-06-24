from optax import Schedule, join_schedules, linear_schedule, chain, trace, identity, add_decayed_weights, scale, scale_by_schedule


def warmup_linear_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
) -> Schedule:
    """Linear warmup followed by Linear decay.

    Args:
        init_value: Initial value for the scalar to be annealed.
        peak_value: Peak value for scalar to be annealed at end of warmup.
        warmup_steps: Positive integer, the length of the linear warmup.
        decay_steps: Positive integer, the total length of the schedule. Note that
            this includes the warmup time, so the number of steps during which Linear
            annealing is applied is `decay_steps - warmup_steps`.
        end_value: End value of the scalar to be annealed.
    Returns:
        schedule: A function that maps step counts to values.
    """
    schedules = [
        linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps),
        linear_schedule(
            init_value=peak_value,
            end_value=end_value,
            transition_steps=decay_steps - warmup_steps),
    ]
    return join_schedules(schedules, [warmup_steps])


# Copy from optax._src.alias
def _scale_by_learning_rate(learning_rate, flip_sign=True):
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return scale_by_schedule(lambda count: m * learning_rate(count))
    return scale(m * learning_rate)


# Copy from optax._src.adamw and sgd
def sgdw(
    learning_rate,
    momentum=None,
    nesterov=False,
    accumulator_dtype=None,
    weight_decay=None,
    mask=None,
):
    return chain(
        (trace(decay=momentum, nesterov=nesterov, accumulator_dtype=accumulator_dtype)
         if momentum is not None else identity()),
        (add_decayed_weights(weight_decay, mask) if weight_decay else identity()),
        _scale_by_learning_rate(learning_rate)
  )
