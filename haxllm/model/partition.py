import re
from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import PartitionSpec as P
import jax
from jax.sharding import NamedSharding


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        n = len(val.shape)
        for rule, replacement in rules:
            if _match(rule, key):
                assert len(replacement) <= n, "Mismatched partition spec {}: {} vs {}" \
                    .format(key, replacement, val.shape)
                if len(replacement) < n:
                    # pad left with None for scan
                    replacement = [None] * (n - len(replacement)) + list(replacement)
                    replacement = P(*replacement)
                return replacement
        # undefined keys in rules are not partitioned
        return P(*[None for _ in range(n)])

    return replace


def get_partition_spec(in_dict, rules):
    replace = _replacement_rules(rules)
    result = {k: replace(k, v) for k, v in flatten_dict(in_dict).items()}
    return freeze(unflatten_dict(result))


def _get_partition_rules_gpt2():
    spec = [
        (("wte", "embedding"), P(None, "Y")),
        (("attn", "(query|key|value)", "kernel"), P("X", "Y", None)), 
        (("attn", "out", "kernel"), P("Y", None, "X")), 
        (("mlp", "fc_1", "kernel"), P("X", "Y")),
        (("mlp", "fc_2", "kernel"), P("Y", "X")),
    ]
    return spec


def _get_partition_rules_llama():
    spec = [
        (("wte", "embedding"), P(None, "Y")),
        (("attn", "(query|key|value)", "kernel"), P("X", "Y", None)), 
        (("attn", "out", "kernel"), P("Y", None, "X")), 
        (("mlp", "gate", "kernel"), P("X", "Y")),
        (("mlp", "up", "kernel"), P("X", "Y")),
        (("mlp", "down", "kernel"), P("Y", "X")),
    ]
    return spec


def get_param_partition_spec(params, model_name):
    if "gpt2" in model_name:
        return get_partition_spec(params, _get_partition_rules_gpt2())
    elif 'llama' in model_name:
        return get_partition_spec(params, _get_partition_rules_llama())
    else:
        raise NotImplementedError


def global_mesh_defined():
    """Checks if global xmap/pjit mesh resource environment is defined."""
    maps_env = jax.experimental.maps.thread_resources.env
    return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


def with_sharding_constraint(x, axis_resources):
    """Wrapper for pjit with_sharding_constraint, no-op on cpu or outside pjit."""
    if jax.devices()[0].platform == 'cpu' or not global_mesh_defined():
        return x
    else:
        return jax.lax.with_sharding_constraint(x, axis_resources)


def with_named_sharding_constraint(x, mesh, partition_spec):
    if mesh is not None:
        return with_sharding_constraint(x, NamedSharding(mesh, partition_spec))
    return x