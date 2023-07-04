import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.modules import RMSNorm
from haxllm.model.parallel import GLUMlpBlock, DenseGeneral
from haxllm.model.utils import load_config as _load_config
from haxllm.model.chatglm2 import (
    config_hub,
    remap_state_dict,
    TransformerConfig as BaseTransformerConfig,
    TransformerModel
)
from haxllm.model.ptuning.modules import PrefixEmbed, SelfAttention


def load_config(name, **kwargs):
    if name in config_hub:
        config = config_hub[name]
    else:
        raise ValueError(f"Unknown chatglm2 model {name}")
    return _load_config(TransformerConfig, config, **kwargs)


@struct.dataclass
class TransformerConfig(BaseTransformerConfig):
    pre_seq_len: int = 128
    prefix_projection: bool = False
    prefix_hidden_size: int = 512
    zero_init_prefix_attn: bool = True


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

        prefix_key_value = PrefixEmbed(
            seq_len=config.pre_seq_len,
            projection=config.prefix_projection,
            prefix_features=config.prefix_hidden_size,
            features=(config.n_heads, config.hidden_size // config.n_heads),
            dtype=config.dtype,
            name="prefix",
        )(inputs)

        if config.memory_efficient_attention or config.decode:
            mask = None
        else:
            mask = nn.make_causal_mask(inputs[..., 0], dtype=jnp.bool_)

        x = RMSNorm(epsilon=config.rms_norm_eps,
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
            zero_init=config.zero_init_prefix_attn,
            name="attn")(x, mask, padding_mask, prefix_key_value=prefix_key_value)
        x = x + inputs

        y = RMSNorm(epsilon=config.rms_norm_eps,
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



class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(
            config=config, block_cls=TransformerBlock, name="transformer")(inputs, train)

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
            config=config, block_cls=TransformerBlock, name="transformer")(inputs=input_ids, train=train)

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
