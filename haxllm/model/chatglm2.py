from typing import Callable, Any

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.modules import RMSNorm, make_block_stack
from haxllm.model.parallel import GLUMlpBlock, DenseGeneral, Embed, SelfAttention
from haxllm.model.utils import load_config as _load_config
from haxllm.config_utils import RematScanConfigMixin


config_hub = {
    "chatglm2-t": dict(
        hidden_size=1024,
        intermediate_size=3456,
        n_heads=8,
        n_layers=2,
        num_groups=2,
    ),
    "chatglm2-6b": dict(
        hidden_size=4096,
        intermediate_size=13696,
        n_heads=32,
        n_layers=28,
        num_groups=2,
    ),
}


def load_config(name, **kwargs):
    if name in config_hub:
        config = config_hub[name]
    else:
        raise ValueError(f"Unknown chatglm2 model {name}")
    return _load_config(TransformerConfig, config, **kwargs)


@struct.dataclass
class TransformerConfig(RematScanConfigMixin):
    vocab_size: int = 65024
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 4096
    intermediate_size: int = 13696
    n_heads: int = 32
    n_layers: int = 28
    num_groups: int = 2
    layer_norm_epsilon: float = 1e-5
    n_positions: int = 32768
    pad_token_id: int = 2
    bos_token_id: int = 1
    eos_token_id: int = 2
    mask_token_id: int = 64789
    gmask_token_id: int = 64790
    memory_efficient_attention: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    decode: bool = False
    shard: bool = False


# TODO: skip apply_query_key_layer_scaling, same as query_key_layer_scaling_coeff in chatglm
# no difference in inference (forward), but may be in training (backward)
class TransformerBlock(nn.Module):
    config: TransformerConfig
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        if isinstance(inputs, tuple):
            inputs, padding_mask = inputs
        else:
            padding_mask = None

        config = self.config

        if config.memory_efficient_attention or config.decode:
            mask = None
        else:
            mask = nn.make_causal_mask(inputs[..., 0], dtype=jnp.bool_)

        x = RMSNorm(epsilon=config.layer_norm_epsilon,
                    dtype=config.dtype, name="ln_1")(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            multi_query_groups=config.num_groups,
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            qkv_bias=True,
            out_bias=False,
            decode=config.decode,
            rope=True,
            query_shard_axes=("X", "Y", None),
            kv_shard_axes=("X", None, "Y"),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            name="attn")(x, mask, padding_mask)

        x = x + inputs

        y = RMSNorm(epsilon=config.layer_norm_epsilon,
                    dtype=config.dtype, name="ln_2")(x)
        y = GLUMlpBlock(
            intermediate_size=config.intermediate_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            use_bias=False,
            shard_axes1=("X", "Y"),
            shard_axes2=("Y", "X"),
            shard=config.shard,
            name="mlp")(y)

        y = x + y

        if padding_mask is not None:
            y = (y, padding_mask)
        if self.scan:
            return y, None
        else:
            return y


class TransformerModel(nn.Module):
    config: TransformerConfig
    block_cls: Callable = TransformerBlock

    @nn.compact
    def __call__(self, inputs, train):
        config = self.config
        remat = config.remat or config.remat_scan

        embed_layer = Embed if remat else Embed
        x = embed_layer(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": (None, "Y")},
            shard=config.shard,
            name="wte")(inputs)

        if config.decode and inputs.shape[1] > 1:
            padding_mask = jnp.equal(inputs, config.pad_token_id)
            x = make_block_stack(
                self.block_cls, config.n_layers, config)((x, padding_mask), train)[0]
        else:
            x = make_block_stack(
                self.block_cls, config.n_layers, config)(x, train)

        norm_layer = RMSNorm if remat else RMSNorm
        x = norm_layer(epsilon=config.layer_norm_epsilon,
                       dtype=config.dtype, name="ln_f")(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(
            config=config, name="transformer")(inputs, train)

        batch_size = inputs.shape[0]
        seq_len = jnp.not_equal(inputs, config.pad_token_id).sum(-1) - 1
        x = x[jnp.arange(batch_size), seq_len]

        x = DenseGeneral(
            config.num_labels,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name="score")(x)
        return x


class TransformerLMHeadModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, input_ids, train=False):
        config = self.config
        x = TransformerModel(
            config=config, name="transformer")(inputs=input_ids, train=train)

        x = DenseGeneral(
            config.vocab_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            kernel_init=config.kernel_init,
            shard_axes={'kernel': ('Y', None)},
            shard=config.shard,
            name="lm_head")(x)
        return x


def remap_state_dict(state_dict):
    state_dict = {k.replace("transformer.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

    n_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith("layers.")]) + 1
    hidden_size = state_dict['embedding.word_embeddings.weight'].shape[1]
    # hard code for now
    head_dim = 128
    num_groups = 2
    n_heads = hidden_size  // head_dim

    root = {}
    root["wte"] = {"embedding": state_dict.pop("embedding.word_embeddings.weight")}

    # TransformerBlock
    for d in range(n_layers):
        block_d = {}
        block_d["ln_1"] = {
            "scale": state_dict.pop(f"layers.{d}.input_layernorm.weight"),
        }
        c_attn_weight = state_dict[f"layers.{d}.self_attention.query_key_value.weight"].T
        c_attn_weight = c_attn_weight.reshape(hidden_size, n_heads * head_dim + 2 * num_groups * head_dim)
        c_attn_bias = state_dict[f"layers.{d}.self_attention.query_key_value.bias"]
        c_attn_bias = c_attn_bias.reshape(n_heads * head_dim + 2 * num_groups * head_dim)

        start = 0
        end = n_heads * head_dim
        query_kernel = c_attn_weight[..., start:end].reshape(hidden_size, n_heads, head_dim)
        query_bias = c_attn_bias[start:end].reshape(n_heads, head_dim)

        start = end
        end = start + num_groups * head_dim
        key_kernel = c_attn_weight[..., start:end].reshape(hidden_size, num_groups, head_dim)
        key_bias = c_attn_bias[start:end].reshape(num_groups, head_dim)

        start = end
        end = start + num_groups * head_dim
        value_kernel = c_attn_weight[..., start:end].reshape(hidden_size, num_groups, head_dim)
        value_bias = c_attn_bias[start:end].reshape(num_groups, head_dim)


        block_d["attn"] = {
            "query": {
                "kernel": query_kernel,
                "bias": query_bias,
            },
            "key": {
                "kernel": key_kernel,
                "bias": key_bias,
            },
            "value": {
                "kernel": value_kernel,
                "bias": value_bias,
            },
            "out": {
                "kernel": state_dict.pop(f"layers.{d}.self_attention.dense.weight").T.reshape(n_heads, head_dim, hidden_size),
            },
        }
        block_d["ln_2"] = {
            "scale": state_dict.pop(f"layers.{d}.post_attention_layernorm.weight"),
        }

        mlp_weight1 = state_dict[f"layers.{d}.mlp.dense_h_to_4h.weight"].T
        intermediate_size = mlp_weight1.shape[1] // 2
        block_d["mlp"] = {
            "gate": {
                "kernel": mlp_weight1[:, 0:intermediate_size],
            },
            "up": {
                "kernel": mlp_weight1[:, intermediate_size:],
            },
            "down": {
                "kernel": state_dict.pop(f"layers.{d}.mlp.dense_4h_to_h.weight").T,
            },
        }
        root[f"h_{d}"] = block_d

    root["ln_f"] = {
        "scale": state_dict.pop("final_layernorm.weight"),
    }
    root["lm_head"] = {"kernel": state_dict.pop("output_layer.weight").T}
    return root