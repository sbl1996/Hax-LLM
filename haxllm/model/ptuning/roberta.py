import functools

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.parallel import DenseGeneral
from haxllm.model.utils import load_config as _load_config
from haxllm.model.ptuning.bert import (
    TransformerConfig as BertConfig,
    TransformerModel,
    TransformerBlock,
)
from haxllm.model.roberta import (
    config_hub,
    remap_state_dict,
    TransformerConfig as RobertaConfig,
)


@struct.dataclass
class TransformerConfig(RobertaConfig, BertConfig):
    pass


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        offset = config.pad_token_id + 1 + config.pre_seq_len
        position_ids = jnp.arange(
            offset, inputs.shape[-1] + offset, dtype=jnp.int32)[None]
        x = TransformerModel(config=config, block_cls=TransformerBlock, name="transformer")(
            inputs=inputs, attn_mask=attn_mask, position_ids=position_ids, train=train)

        if not config.pooler:
            x = x[:, 0]
        x = nn.Dropout(rate=config.cls_pdrop)(x, not train)

        dense = functools.partial(
            DenseGeneral,
            dtype=config.dtype,
            param_dtype=jnp.float32,
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
