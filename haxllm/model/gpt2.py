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
    "gpt2": dict(
        hidden_size=768,
        n_heads=12,
        n_layers=12,
    ),
    "gpt2-medium": dict(
        hidden_size=1024,
        n_heads=16,
        n_layers=24,
    ),
    "gpt2-large": dict(
        hidden_size=1280,
        n_heads=20,
        n_layers=36,
    ),
    "gpt2-xl": dict(
        hidden_size=1600,
        n_heads=25,
        n_layers=48,
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
    vocab_size: int = 50257
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 768
    n_heads: int = 12
    n_layers: int = 12
    layer_norm_epsilon: float = 1e-5
    n_positions: int = 1024
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    cls_pdrop: float = 0.1
    pad_token_id: int = 50256
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    decode: bool = False
    shard: bool = False


class TransformerBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        if not config.decode:
            mask = nn.make_causal_mask(inputs[..., 0], dtype=jnp.bool_)
        else:
            mask = None

        x = nn.LayerNorm(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_1"
        )(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            dropout_rate=config.attn_pdrop,
            deterministic=self.deterministic,
            decode=config.decode,
            qkv_shard_axes=("X", "Y", None),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            name="attn")(x, mask)
        x = nn.Dropout(rate=config.resid_pdrop)(x, deterministic=self.deterministic)
        x = x + inputs

        y = nn.LayerNorm(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_2"
        )(x)
        y = MlpBlock(
            activation="gelu_new",
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes1=("X", "Y"),
            shard_axes2=("Y", "X"),
            shard=config.shard,
            name="mlp",
        )(y)
        y = nn.Dropout(rate=config.resid_pdrop)(y, deterministic=self.deterministic)

        if self.scan:
            return x + y, None
        else:
            return x + y


class TransformerModel(nn.Module):
    config: TransformerConfig
    block_cls: Callable = TransformerBlock

    @nn.compact
    def __call__(self, *, input_ids, train):
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

        position_ids = jnp.arange(0, input_ids.shape[-1], dtype=jnp.int32)[None]
        if config.decode:
            is_initialized = self.has_variable("cache", "cache_position_ids")
            cache_position_ids = self.variable(
                "cache", "cache_position_ids", lambda: jnp.array(0, dtype=jnp.int32)
            )
            if is_initialized:
                batch_size, seq_len = input_ids.shape[:2]
                if seq_len != 1:
                    start = cache_position_ids.value
                    offset = jnp.argmax(input_ids[0] == config.pad_token_id)
                    offset = jnp.where(offset == 0, seq_len, offset)
                    position_ids = jnp.arange(seq_len, dtype=jnp.int32) + start
                else:
                    start = cache_position_ids.value
                    offset = 1
                    position_ids = cache_position_ids.value[None]
                cache_position_ids.value = start + offset
                position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_len))

        inputs_embeds = embed_layer(num_embeddings=config.vocab_size, name="wte")(input_ids)
        position_embeds = embed_layer(num_embeddings=config.n_positions, name="wpe")(position_ids)

        x = inputs_embeds + position_embeds

        dropout_layer = nn.remat(nn.Dropout, static_argnums=(2,)) if remat else nn.Dropout
        x = dropout_layer(rate=config.embd_pdrop)(x, not train)

        block_fn = functools.partial(self.block_cls, deterministic=not train)
        x = make_block_stack(block_fn, config.n_layers, config)(x, train)

        norm_layer = nn.remat(nn.LayerNorm) if remat else nn.LayerNorm
        x = norm_layer(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_f"
        )(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, input_ids, attn_mask, train=False):
        config = self.config
        x = TransformerModel(config=config, name="transformer")(
            input_ids=input_ids, train=train
        )

        batch_size = input_ids.shape[0]
        seq_len = jnp.not_equal(input_ids, config.pad_token_id).sum(-1) - 1
        x = x[jnp.arange(batch_size), seq_len]

        x = nn.Dropout(rate=config.cls_pdrop)(x, not train)
        x = DenseGeneral(
            config.num_labels,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name="score",
        )(x)
        return x


class TransformerLMHeadModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, input_ids, train=False):
        config = self.config
        x = TransformerModel(config=config, name="transformer")(
            input_ids=input_ids, train=train)

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


def remap_state_dict(state_dict):
    n_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith("h.")]) + 1
    hidden_size = state_dict['wte.weight'].shape[1]
    # hard code for GPT-2
    head_dim = 64
    n_heads = hidden_size  // head_dim

    # Embedding
    root = {}
    root["wte"] = {"embedding": state_dict.pop("wte.weight")}
    root["wpe"] = {"embedding": state_dict.pop("wpe.weight")}

    # TransformerBlock
    for d in range(n_layers):
        block_d = {}
        block_d["ln_1"] = {
            "scale": state_dict.pop(f"h.{d}.ln_1.weight"),
            "bias": state_dict.pop(f"h.{d}.ln_1.bias"),
        }
        c_attn_weight = state_dict[f"h.{d}.attn.c_attn.weight"]
        c_attn_bias = state_dict[f"h.{d}.attn.c_attn.bias"]
        block_d["attn"] = {
            "query": {
                "kernel": c_attn_weight[:, 0:hidden_size].reshape(
                    hidden_size, n_heads, -1
                ),
                "bias": c_attn_bias[0:hidden_size].reshape(n_heads, -1),
            },
            "key": {
                "kernel": c_attn_weight[:, hidden_size: hidden_size * 2].reshape(
                    hidden_size, n_heads, -1
                ),
                "bias": c_attn_bias[hidden_size: hidden_size * 2].reshape(n_heads, -1),
            },
            "value": {
                "kernel": c_attn_weight[:, hidden_size * 2: hidden_size * 3].reshape(
                    hidden_size, n_heads, -1
                ),
                "bias": c_attn_bias[hidden_size * 2: hidden_size * 3].reshape(
                    n_heads, -1
                ),
            },
            "out": {
                "kernel": state_dict.pop(f"h.{d}.attn.c_proj.weight").reshape(
                    n_heads, -1, hidden_size
                ),
                "bias": state_dict.pop(f"h.{d}.attn.c_proj.bias"),
            },
        }
        block_d["ln_2"] = {
            "scale": state_dict.pop(f"h.{d}.ln_2.weight"),
            "bias": state_dict.pop(f"h.{d}.ln_2.bias"),
        }
        block_d["mlp"] = {
            "fc_1": {
                "kernel": state_dict.pop(f"h.{d}.mlp.c_fc.weight"),
                "bias": state_dict.pop(f"h.{d}.mlp.c_fc.bias"),
            },
            "fc_2": {
                "kernel": state_dict.pop(f"h.{d}.mlp.c_proj.weight"),
                "bias": state_dict.pop(f"h.{d}.mlp.c_proj.bias"),
            },
        }
        root[f"h_{d}"] = block_d

    root["ln_f"] = {
        "scale": state_dict.pop("ln_f.weight"),
        "bias": state_dict.pop("ln_f.bias"),
    }
    root["lm_head"] = {"kernel": root["wte"]["embedding"].T}
    return root
