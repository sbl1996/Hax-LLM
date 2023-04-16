import re
from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import PartitionSpec as P
import jax
from jax.sharding import NamedSharding

# Sentinels
_unmatched = object()

# For specifying empty leaf dict `{}`
empty_dict = object()


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
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace

def get_partition_spec(in_dict, rules):
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    for k, v in result.items():
        if v == _unmatched:
            print("Unmatched key: {}".format(k))
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))


def _get_partition_rules_gpt2():
    spec = [
        (("transformer", "wte", "embedding"), P(None, None)),
        (("transformer", "wpe", "embedding"), P(None, None)),

        (("hs", "attn", "(query|key|value)", "kernel"), P("X", "Y", None)), 
        (("hs", "attn", "(query|key|value)", "bias"), P(None, None)), 
        (("hs", "attn", "out", "kernel"), P("Y", None, "X")), 
        (("hs", "attn", "out", "bias"), P(None)), 
        (("hs", "mlp", "fc_1", "kernel"), P("X", "Y")),
        (("hs", "mlp", "fc_1", "bias"), P(None)),
        (("hs", "mlp", "fc_2", "kernel"), P("Y", "X")),
        (("hs", "mlp", "fc_2", "bias"), P(None)), 
        (("hs", "ln_1", "(scale|bias)"), P(None)),
        (("hs", "ln_2", "(scale|bias)"), P(None)),

        (("transformer", "ln_f", "(scale|bias)"), P(None)), 
        (("score", "kernel"), P(None, None)), 
        (("score", "bias"), P(None)), 
    ]
    spec_ = []
    for k, v in spec:
        if k[0] == 'hs':
            v = P(*[None, None, *v])
        spec_.append((k, v))
    return spec_


def get_gpt2_param_partition_spec(params):
    return get_partition_spec(params, _get_partition_rules_gpt2())

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