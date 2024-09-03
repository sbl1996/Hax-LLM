import functools
from typing import Any, Callable

import jax.numpy as jnp

from flax import struct
from flax import linen as nn

from haxllm.model.parallel import SelfAttention, MlpBlock, DenseGeneral
from haxllm.model.modules import make_block_stack
from haxllm.model.utils import load_config as _load_config, truncated_normal_init
from haxllm.model.mixin import RematScanConfigMixin


config_hub = {
    "bert-base-uncased": dict(
        hidden_size=768,
        intermediate_size=3072,
        n_heads=12,
        n_layers=12,
    ),
    "bert-large-uncased": dict(
        hidden_size=1024,
        intermediate_size=4096,
        n_heads=16,
        n_layers=24,
    ),
    "bert-base-cased": dict(
        hidden_size=768,
        intermediate_size=3072,
        n_heads=12,
        n_layers=12,
    ),
    "bert-large-cased": dict(
        hidden_size=1024,
        intermediate_size=4096,
        n_heads=16,
        n_layers=24,
    ),
    "roberta-large": dict(
        hidden_size=1024,
        intermediate_size=4096,
        n_heads=16,
        n_layers=24,
        vocab_size=50265,
    ),
}


@struct.dataclass
class TransformerConfig(RematScanConfigMixin):
    patch_size: int = 16
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 768
    intermediate_size: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    layer_norm_epsilon: float = 1e-6
    n_positions: int = 1024
    initializer_range: float = 0.02
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    cls_pdrop: float = 0.0
    pad_token_id: int = 0
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros_init()
    shard: bool = False


class TransformerBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        inputs, padding_mask = inputs
        config = self.config

        if padding_mask is not None:
            padding_mask_ = padding_mask[:, None, None, :]
        else:
            padding_mask_ = None

        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name="ln_1")(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            dropout_rate=config.attn_pdrop,
            deterministic=self.deterministic,
            broadcast_dropout=False,
            query_shard_axes=("X", "Y", None),
            kv_shard_axes=("X", "Y", None),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            name="attn")(x, padding_mask=padding_mask_)
        x = nn.Dropout(rate=config.resid_pdrop)(
            x, deterministic=self.deterministic)
        x = x + inputs

        y = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name="ln_2")(x)
        y = MlpBlock(
            intermediate_size=config.intermediate_size,
            activation="gelu_new",
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            shard_axes1=("X", "Y"),
            shard_axes2=("Y", "X"),
            shard=config.shard,
            name="mlp")(y)
        y = nn.Dropout(rate=config.resid_pdrop)(
            y, deterministic=self.deterministic)

        y = x + y

        outputs = (y, padding_mask)
        if self.scan:
            return outputs, None
        else:
            return outputs


class TransformerModel(nn.Module):
    config: TransformerConfig
    block_cls: Callable = TransformerBlock

    @nn.compact
    def __call__(self, *, inputs, attn_mask, position_ids=None, train=False):
        config = self.config

        if position_ids is None:
            position_ids = jnp.arange(
                0, inputs.shape[-1], dtype=jnp.int32)[None]

        patch_embeds = nn.Conv(
            config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_size, config.patch_size),
            padding="VALID",
            dtype=config.dtype,
            param_dtype=config.param_dtype,
        )(inputs)
        position_embeds = nn.Embed(
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            num_embeddings=config.n_positions,
            name="position_embeddings"
        )(position_ids)

        x = patch_embeds + position_embeds

        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon,
                         dtype=config.dtype, name="ln")(x)

        x = nn.Dropout(rate=config.embd_pdrop)(x, not train)

        block_fn = functools.partial(
            self.block_cls, deterministic=not train)
        x = make_block_stack(block_fn, config.n_layers,
                             config)((x, attn_mask), train)[0]

        if config.pooler:
            x = DenseGeneral(
                config.hidden_size,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
                kernel_init=truncated_normal_init(
                    stddev=config.initializer_range),
                bias_init=config.bias_init,
                name="pooler_fc")(x[:, 0])
            x = jnp.tanh(x)
        return x


def remap_state_dict(state_dict, head_dim=None):
    state_dict = {k.replace("bert.", ""): v for k, v in state_dict.items()}

    n_layers = max([int(k.split('.')[2]) for k in state_dict.keys() if k.startswith("encoder.layer.")]) + 1
    hidden_size = state_dict['embeddings.word_embeddings.weight'].shape[1]
    # hard code for bert
    if head_dim is None:
        head_dim = 64
    n_heads = hidden_size  // head_dim

    root = {}
    root["word_embeddings"] = {"embedding": state_dict.pop(
        "embeddings.word_embeddings.weight")}
    root["position_embeddings"] = {"embedding": state_dict.pop(
        "embeddings.position_embeddings.weight")}
    root["token_type_embeddings"] = {"embedding": state_dict.pop(
        "embeddings.token_type_embeddings.weight")}
    root["ln"] = {"scale": state_dict.pop(
        "embeddings.LayerNorm.gamma"), "bias": state_dict.pop("embeddings.LayerNorm.beta")}

    for d in range(n_layers):
        block_d = {}
        block_d["attn"] = {
            "query": {
                "kernel": state_dict.pop(f"encoder.layer.{d}.attention.self.query.weight").T.reshape(hidden_size, n_heads, -1),
                "bias": state_dict.pop(f"encoder.layer.{d}.attention.self.query.bias").reshape(n_heads, -1),
            },
            "key": {
                "kernel": state_dict.pop(f"encoder.layer.{d}.attention.self.key.weight").T.reshape(hidden_size, n_heads, -1),
                "bias": state_dict.pop(f"encoder.layer.{d}.attention.self.key.bias").reshape(n_heads, -1),
            },
            "value": {
                "kernel": state_dict.pop(f"encoder.layer.{d}.attention.self.value.weight").T.reshape(hidden_size, n_heads, -1),
                "bias": state_dict.pop(f"encoder.layer.{d}.attention.self.value.bias").reshape(n_heads, -1),
            },
            "out": {
                "kernel": state_dict.pop(f"encoder.layer.{d}.attention.output.dense.weight").T.reshape(n_heads, -1, hidden_size),
                "bias": state_dict.pop(f"encoder.layer.{d}.attention.output.dense.bias"),
            },
        }
        block_d["ln_1"] = {"scale": state_dict.pop(f"encoder.layer.{d}.attention.output.LayerNorm.gamma"),
                           "bias": state_dict.pop(f"encoder.layer.{d}.attention.output.LayerNorm.beta")}

        block_d["mlp"] = {
            "fc_1": {
                "kernel": state_dict.pop(f"encoder.layer.{d}.intermediate.dense.weight").T,
                "bias": state_dict.pop(f"encoder.layer.{d}.intermediate.dense.bias"),
            },
            "fc_2": {
                "kernel": state_dict.pop(f"encoder.layer.{d}.output.dense.weight").T,
                "bias": state_dict.pop(f"encoder.layer.{d}.output.dense.bias"),
            },
        }
        block_d["ln_2"] = {"scale": state_dict.pop(f"encoder.layer.{d}.output.LayerNorm.gamma"),
                           "bias": state_dict.pop(f"encoder.layer.{d}.output.LayerNorm.beta")}
        root[f"h_{d}"] = block_d

    root["pooler_fc"] = {"kernel": state_dict.pop(
        "pooler.dense.weight").T, "bias": state_dict.pop("pooler.dense.bias")}

    return root
