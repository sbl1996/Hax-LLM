from optax import Schedule, join_schedules, linear_schedule


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

  