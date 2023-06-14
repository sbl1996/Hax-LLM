import functools
from typing import Any, Callable

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.parallel import SelfAttention, DenseGeneral, Embed, MlpBlock
from haxllm.model.utils import load_config as _load_config
from haxllm.model.modules import make_block_stack
from haxllm.model.chatglm import (
    config_hub,
    remap_state_dict,
    TransformerConfig as BaseTransformerConfig,
    get_masks, get_position_ids,
)
from haxllm.model.ptuning.modules import PrefixEmbed, SelfAttention


def load_config(name, **kwargs):
    if name in config_hub:
        config = config_hub[name]
    else:
        raise ValueError(f"Unknown gpt2 model {name}")
    return _load_config(TransformerConfig, config, **kwargs)


@struct.dataclass
class TransformerConfig(BaseTransformerConfig):
    pre_seq_len: int = 0
    prefix_projection: bool = False
    prefix_hidden_size: int = 512
    zero_init_prefix_attn: bool = False


# TODO: skip query_key_layer_scaling_coeff
# no difference in inference (forward), but may be in training (backward)
class TransformerBlock(nn.Module):
    config: TransformerConfig
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        x, position_ids = inputs

        mask = get_masks(position_ids) if not config.decode else None

        attn_input = nn.LayerNorm(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_1")(x)
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

        alpha = (2 * config.n_layers) ** 0.5
        x = attn_input * alpha + attn_output

        mlp_input = nn.LayerNorm(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_2")(x)
        mlp_output = MlpBlock(
            activation="gelu_new",
            use_bias=True,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes1=("X", "Y"),
            shard_axes2=("Y", "X"),
            shard=config.shard,
            name="mlp")(mlp_input)

        y = mlp_input * alpha + mlp_output
        if self.scan:
            return (y, position_ids), None
        else:
            return y, position_ids


class TransformerModel(nn.Module):
    config: TransformerConfig
    block_cls: Callable = TransformerBlock

    @nn.compact
    def __call__(self, *, input_ids, train):
        config = self.config
        remat = config.remat or config.remat_scan

        position_ids = get_position_ids(input_ids, config.bos_token_id, config.mask_token_id, config.gmask_token_id)
        if config.decode:
            is_initialized = self.has_variable("cache", "cache_position_ids")
            cache_position_ids = self.variable(
                "cache", "cache_position_ids", lambda: jnp.array([0, 0], dtype=jnp.uint32)
            )
            if is_initialized:
                batch_size, seq_len = input_ids.shape
                # See get_position_ids for details
                if seq_len > 1:
                    # First stage of ChatGLM decoding
                    mask_position = jnp.argmax(input_ids[0] == config.mask_token_id)
                    gmask_position = jnp.argmax(input_ids[0] == config.gmask_token_id)
                    mask_position = jnp.where(mask_position == 0, gmask_position, mask_position)
                    cache_position_ids.value = jnp.array([mask_position, 1], dtype=jnp.uint32)
                else:
                    p = cache_position_ids.value
                    p = p.at[1].set(p[1] + 1)
                    position_ids = jnp.broadcast_to(p[None, :, None], (batch_size, 2, 1))
                    cache_position_ids.value = p

        embed_layer = nn.remat(Embed) if remat else Embed
        embed_layer = functools.partial(
            embed_layer,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": (None, "Y")},
            shard=config.shard,
        )

        x = embed_layer(num_embeddings=config.vocab_size, name="wte")(input_ids)


        block_fn = self.block_cls
        x = make_block_stack(block_fn, config.n_layers, config)((x, position_ids), train)[0]

        norm_layer = nn.remat(nn.LayerNorm) if remat else nn.LayerNorm
        x = norm_layer(
            epsilon=config.layer_norm_epsilon, dtype=config.dtype, name="ln_f"
        )(x)
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
