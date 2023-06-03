import functools

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.parallel import DenseGeneral
from haxllm.model.utils import load_config as _load_config
from haxllm.model.bert import TransformerConfig as BertConfig, TransformerModel


config_hub = {
    "roberta-large": dict(
        hidden_size=1024,
        intermediate_size=4096,
        n_heads=16,
        n_layers=24,
    ),
}


def load_config(name, **kwargs):
    if name in config_hub:
        config = config_hub[name]
    else:
        raise ValueError(f"Unknown roberta model {name}")
    return _load_config(TransformerConfig, config, **kwargs)


@struct.dataclass
class TransformerConfig(BertConfig):
    vocab_size: int = 50265
    type_vocab_size: int = 1
    n_positions: int = 514
    layer_norm_epsilon: float = 1e-5
    pooler: bool = False
    bos_token_id: int = 0
    pad_token_id: int = 1
    eos_token_id: int = 2


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        offset = config.pad_token_id + 1
        position_ids = jnp.arange(
            offset, inputs.shape[-1] + offset, dtype=jnp.int32)[None]
        x = TransformerModel(config=config, name="transformer")(
            inputs=inputs, attn_mask=attn_mask, position_ids=position_ids, train=train)

        if not config.pooler:
            x = x[:, 0]
        x = nn.Dropout(rate=config.cls_pdrop)(x, not train)

        dense = functools.partial(
            DenseGeneral,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )

        x = dense(config.hidden_size, name="cls_dense")(x)
        x = jnp.tanh(x)
        x = nn.Dropout(rate=config.cls_pdrop)(x, not train)
        x = dense(
            config.num_labels,
            kernel_init=nn.initializers.normal(
                stddev=config.initializer_range),
            name="score")(x)
        return x


def remap_state_dict(state_dict, config: TransformerConfig):
    state_dict = {k.replace("roberta.", ""): v for k, v in state_dict.items()}
    root = {}
    root["word_embeddings"] = {"embedding": state_dict.pop(
        "embeddings.word_embeddings.weight")}
    root["position_embeddings"] = {"embedding": state_dict.pop(
        "embeddings.position_embeddings.weight")}
    root["token_type_embeddings"] = {"embedding": state_dict.pop(
        "embeddings.token_type_embeddings.weight")}
    root["ln"] = {"scale": state_dict.pop(
        "embeddings.LayerNorm.weight"), "bias": state_dict.pop("embeddings.LayerNorm.bias")}
    hidden_size = config.hidden_size
    n_heads = config.n_heads

    for d in range(config.n_layers):
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
        block_d["ln_1"] = {"scale": state_dict.pop(f"encoder.layer.{d}.attention.output.LayerNorm.weight"),
                           "bias": state_dict.pop(f"encoder.layer.{d}.attention.output.LayerNorm.bias")}

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
        block_d["ln_2"] = {"scale": state_dict.pop(f"encoder.layer.{d}.output.LayerNorm.weight"),
                           "bias": state_dict.pop(f"encoder.layer.{d}.output.LayerNorm.bias")}
        root[f"h_{d}"] = block_d
    return root
