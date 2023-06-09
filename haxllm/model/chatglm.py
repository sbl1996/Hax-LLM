import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.parallel import SelfAttention, DenseGeneral, Embed, MlpBlock
from haxllm.model.utils import load_config as _load_config
from haxllm.model.modules import make_block_stack
from haxllm.config_utils import RematScanConfigMixin


config_hub = {
    "chatglm-6b": dict(
        hidden_size=4096,
        n_heads=32,
        n_layers=28,
        intermediate_size=16384,
    ),
}


def load_config(name, **kwargs):
    if name in config_hub:
        config = config_hub[name]
    else:
        raise ValueError(f"Unknown gpt2 model {name}")
    return _load_config(TransformerConfig, config, **kwargs)


@struct.dataclass
class TransformerConfig(RematScanConfigMixin):
    vocab_size: int = 130528
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 4096
    intermediate_size: int = 16384
    n_heads: int = 32
    n_layers: int = 28
    layer_norm_epsilon: float = 1e-5
    n_positions: int = 2048
    pad_token_id: int = 3
    bos_token_id: int = 130004
    eos_token_id: int = 130005
    mask_token_id: int = 130000,
    gmask_token_id: int = 130001,
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    decode: bool = False
    shard: bool = False


# TODO: skip query_key_layer_scaling_coeff
# no difference in inference (forward), but may be in training (backward)
class TransformerBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        x, position_ids = inputs

        seq_len = x.shape[1]
        if not config.decode:
            # training
            idxs = jnp.arange(seq_len, dtype=jnp.int32)
            # autually context_lengths - 1
            context_lengths = position_ids[:, :1, -1:]
            idxs1 = idxs[None, :, None]
            idxs2 = idxs[None, None, :]
            mask = (idxs1 >= idxs2) | (idxs2 <= context_lengths)
            mask = mask[:, None, :, :]
        else:
            # decode
            mask = None

        attn_input = nn.LayerNorm(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_1")(x)
        # jax.debug.print("attn_input {}", attn_input[0, :, :10])
        attn_output = SelfAttention(
            num_heads=config.n_heads,
            max_len=config.n_positions,
            use_bias=True,
            rope=True,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            decode=config.decode,
            qkv_shard_axes=("X", "Y", None),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            name="attn")(attn_input, mask=mask, position_ids=position_ids)
        # jax.debug.print("attn_output {}", attn_output[0, :, :10])

        alpha = (2 * config.n_layers) ** 0.5
        x = attn_input * alpha + attn_output
        # jax.debug.print("res1 {}", x[0, :, :10])

        mlp_input = nn.LayerNorm(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_2")(x)
        # jax.debug.print("mlp_input {}", mlp_input[0, :, :10])
        mlp_output = MlpBlock(
            activation="gelu_new",
            use_bias=True,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes1=("X", "Y"),
            shard_axes2=("Y", "X"),
            shard=config.shard,
            name="mlp")(mlp_input)
        # jax.debug.print("mlp_output {}", mlp_output[0, :, :10])

        y = mlp_input * alpha + mlp_output
        # jax.debug.print("res2 {}", y[0, :, :10])
        if self.scan:
            return (y, position_ids), None
        else:
            return y, position_ids


class TransformerModel(nn.Module):
    config: TransformerConfig
    block_cls: Callable = TransformerBlock

    @nn.compact
    def __call__(self, *, inputs, position_ids, train):
        config = self.config
        remat = config.remat or config.remat_scan

        embed_layer = nn.remat(Embed) if remat else Embed
        embed_layer = functools.partial(
            embed_layer,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": (None, "Y")},
            shard=config.shard,
        )

        x = embed_layer(num_embeddings=config.vocab_size, name="wte")(inputs)

        # jax.debug.print("embed {}", x[0, :, :10])

        block_fn = functools.partial(self.block_cls, deterministic=not train)
        x = make_block_stack(block_fn, config.n_layers, config)((x, position_ids), train)[0]

        norm_layer = nn.remat(nn.LayerNorm) if remat else nn.LayerNorm
        x = norm_layer(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_f"
        )(x)
        # jax.debug.print("out {}", x[0, :, :10])
        return x


# class TransformerSequenceClassifier(nn.Module):
#     config: TransformerConfig

#     @nn.compact
#     def __call__(self, *, inputs, attn_mask, train=False):
#         config = self.config
#         x = TransformerModel(config=config, name="transformer")(
#             inputs=inputs, train=train
#         )

#         batch_size = inputs.shape[0]
#         seq_len = jnp.not_equal(inputs, config.pad_token_id).sum(-1) - 1
#         x = x[jnp.arange(batch_size), seq_len]

#         x = nn.Dropout(rate=config.cls_pdrop)(x, not train)
#         x = DenseGeneral(
#             config.num_labels,
#             dtype=config.dtype,
#             kernel_init=config.kernel_init,
#             bias_init=config.bias_init,
#             name="score",
#         )(x)
#         return x


class TransformerLMHeadModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, input_ids, position_ids, train=False):
        config = self.config
        x = TransformerModel(config=config, name="transformer")(
            inputs=input_ids, position_ids=position_ids, train=train)

        x = DenseGeneral(
            config.vocab_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            kernel_init=config.kernel_init,
            shard_axes={'kernel': ('Y', None)},
            shard=config.shard,
            name="lm_head",
        )(x)
        return x


def remap_state_dict(state_dict, config: TransformerConfig):
    state_dict = {k.replace("transformer.", ""): v for k,
                  v in state_dict.items()}
    root = {}
    root["wte"] = {"embedding": state_dict.pop("word_embeddings.weight")}
    hidden_size = config.hidden_size
    n_heads = config.n_heads

    # TransformerBlock
    for d in range(config.n_layers):
        block_d = {}
        block_d["ln_1"] = {
            "scale": state_dict.pop(f"layers.{d}.input_layernorm.weight"),
            "bias": state_dict.pop(f"layers.{d}.input_layernorm.bias"),
        }
        c_attn_weight = state_dict[f"layers.{d}.attention.query_key_value.weight"].T
        c_attn_weight = c_attn_weight.reshape(hidden_size, n_heads, -1)
        head_dim = c_attn_weight.shape[-1] // 3
        c_attn_bias = state_dict[f"layers.{d}.attention.query_key_value.bias"]
        c_attn_bias = c_attn_bias.reshape(n_heads, -1)
        block_d["attn"] = {
            "query": {
                "kernel": c_attn_weight[..., 0:head_dim],
                "bias": c_attn_bias[:, 0:head_dim],
            },
            "key": {
                "kernel": c_attn_weight[..., head_dim: head_dim * 2],
                "bias": c_attn_bias[:, head_dim: head_dim * 2].reshape(n_heads, -1),
            },
            "value": {
                "kernel": c_attn_weight[..., head_dim * 2: head_dim * 3],
                "bias": c_attn_bias[:, head_dim * 2: head_dim * 3].reshape(n_heads, -1),
            },
            "out": {
                "kernel": state_dict.pop(f"layers.{d}.attention.dense.weight").T.reshape(n_heads, -1, hidden_size),
                "bias": state_dict.pop(f"layers.{d}.attention.dense.bias"),
            },
        }
        block_d["ln_2"] = {
            "scale": state_dict.pop(f"layers.{d}.post_attention_layernorm.weight"),
            "bias": state_dict.pop(f"layers.{d}.post_attention_layernorm.bias"),
        }
        block_d["mlp"] = {
            "fc_1": {
                "kernel": state_dict.pop(f"layers.{d}.mlp.dense_h_to_4h.weight").T,
                "bias": state_dict.pop(f"layers.{d}.mlp.dense_h_to_4h.bias"),
            },
            "fc_2": {
                "kernel": state_dict.pop(f"layers.{d}.mlp.dense_4h_to_h.weight").T,
                "bias": state_dict.pop(f"layers.{d}.mlp.dense_4h_to_h.bias"),
            },
        }
        root[f"h_{d}"] = block_d

    root["ln_f"] = {
        "scale": state_dict.pop("final_layernorm.weight"),
        "bias": state_dict.pop("final_layernorm.bias"),
    }
    root["lm_head"] = {"kernel": state_dict.pop("lm_head.weight").T}
    return root
