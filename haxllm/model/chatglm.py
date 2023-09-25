# deprecated

import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.parallel import SelfAttention, DenseGeneral, Embed, MlpBlock
from haxllm.model.utils import load_config as _load_config
from haxllm.model.modules import make_block_stack
from haxllm.model.mixin import RematScanConfigMixin


config_hub = {
    "chatglm-t": dict(
        hidden_size=1024,
        n_heads=8,
        n_layers=2,
        intermediate_size=4096,
    ),
    "chatglm-6b": dict(
        hidden_size=4096,
        n_heads=32,
        n_layers=28,
        intermediate_size=16384,
    ),
}


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
    mask_token_id: int = 130000
    gmask_token_id: int = 130001
    memory_efficient_attention: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    decode: bool = False
    shard: bool = False


# Example of ChatGLM position_ids and masks
# sentence = "你好"
# input_ids:
# [5, 74874, 130001, 130004]
# where 130001 is gmask_token_id and 130004 is bos_token_id
# 
# position_ids:
# [[0, 1, 2, 2],
#  [0, 0, 0, 1]]
# 
# attention_mask:
# [[False, False, False,  True],
#  [False, False, False,  True],
#  [False, False, False,  True],
#  [False, False, False, False]]
# 
# From the above example, we define the following:
#  context_length = 3, mask_position = 2


def get_position_ids(input_ids, bos_token_id, mask_token_id, gmask_token_id):
    # Translated from: THUDM/chatglm-6b, tokenization_chatglm.py

    # We assume that the bos token is always in the input
    # if bos_token_id in required_input:
    #   context_length = required_input.index(bos_token_id)
    # else:
    #   context_length = seq_length
    context_lengths = jnp.argmax(input_ids == bos_token_id, axis=1, keepdims=True)

    # Translated from:
    # mask_token = mask_token_id if mask_token_id in required_input else gmask_token_id
    mask_position = jnp.argmax(input_ids == mask_token_id, axis=1, keepdims=True)
    gmask_position = jnp.argmax(input_ids == gmask_token_id, axis=1, keepdims=True)
    mask_position = jnp.where(mask_position == 0, gmask_position, mask_position)
    
    # We skip it, asuming that the mask token is always in the input
    # if mask_token in required_input:

    # Translated from:
    #   mask_position = required_input.index(mask_token)
    #   position_ids[context_length:] = mask_position
    seq_len = input_ids.shape[1]
    B = jnp.arange(0, seq_len, dtype=jnp.int32)[None, :]
    position_ids = jnp.where(B < context_lengths, B, mask_position)

    # Translated from:
    # block_position_ids = np.concatenate(
    #   [np.zeros(context_length, dtype=np.int64),
    #   np.arange(1, seq_length - context_length + 1, dtype=np.int64)])
    # encoded_inputs["position_ids"] = np.stack([position_ids, block_position_ids], axis=0)
    block_position_ids = jnp.maximum(B - context_lengths + 1, 0)
    position_ids = jnp.stack([position_ids, block_position_ids], axis=1)  
    return position_ids      


def make_bidirectional_mask(context_lengths, seq_len):
    # Translated from:
    # attention_mask = np.ones((1, seq_length, seq_length))
    # attention_mask = np.tril(attention_mask)
    # attention_mask[:, :, :context_length] = 1
    # attention_mask = np.bool_(attention_mask < 0.5) 
    idxs = jnp.arange(seq_len, dtype=jnp.int32)
    idxs1 = idxs[None, :, None]
    idxs2 = idxs[None, None, :]
    mask = (idxs1 >= idxs2) | (idxs2 < context_lengths[:, None, None])
    mask = mask[:, None, :, :]  # (batch_size, 1, seq_len, seq_len)
    return mask


def get_masks(position_ids):
    # Translated from: THUDM/chatglm-6b, tokenization_chatglm.py

    # We assume that the bos token is always in the input
    # ```
    # if bos_token_id in required_input:
    #   context_length = required_input.index(bos_token_id)
    # else:
    #   context_length = seq_length
    # ```
    # We don't have input_ids here, so we infer it from position_ids
    # See: position_ids = jnp.where(B < context_lengths, B, mask_position)
    context_lengths = jnp.argmax(position_ids[:, 0, :], axis=1) + 1
    seq_len = position_ids.shape[-1]
    return make_bidirectional_mask(context_lengths, seq_len)


# TODO: skip query_key_layer_scaling_coeff
# no difference in inference (forward), but may be in training (backward)
class TransformerBlock(nn.Module):
    config: TransformerConfig
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        x, position_ids = inputs

        if config.decode or config.memory_efficient_attention:
            mask = None
        else:
            mask = get_masks(position_ids)

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
            memory_efficient=config.memory_efficient_attention,
            memory_efficient_mask_mode='bidirectional',
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

        if config.decode:
            shard_axes = {"kernel": ("Y", None)}
        else:
            # shard output in training to avoid out of memory
            shard_axes = {'kernel': (None, 'Y')}
        x = DenseGeneral(
            config.vocab_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=jnp.float32,
            kernel_init=config.kernel_init,
            shard_axes=shard_axes,
            shard=config.shard,
            name="lm_head",
        )(x)
        return x


def remap_state_dict(state_dict):
    state_dict = {k.replace("transformer.", ""): v for k, v in state_dict.items()}

    n_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith("layers.")]) + 1
    hidden_size = state_dict['word_embeddings.weight'].shape[1]
    # hard code for now
    head_dim = 128
    n_heads = hidden_size  // head_dim

    root = {}
    root["wte"] = {"embedding": state_dict.pop("word_embeddings.weight")}

    # TransformerBlock
    for d in range(n_layers):
        block_d = {}
        block_d["ln_1"] = {
            "scale": state_dict.pop(f"layers.{d}.input_layernorm.weight"),
            "bias": state_dict.pop(f"layers.{d}.input_layernorm.bias"),
        }
        c_attn_weight = state_dict[f"layers.{d}.attention.query_key_value.weight"].T
        c_attn_weight = c_attn_weight.reshape(hidden_size, n_heads, head_dim * 3)
        c_attn_bias = state_dict[f"layers.{d}.attention.query_key_value.bias"]
        c_attn_bias = c_attn_bias.reshape(n_heads, head_dim * 3)
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


class ChatSetting:
    name = "chatglm"
    system = ""
    roles = ("问", "答")
    stop_token_ids = (0, 2)

    def get_prompt(self, messages):
        sep = "\n\n"
        if self.system:
            ret = self.system + sep
        else:
            ret = ""
        for i, (role, message) in enumerate(messages):
            if i % 2 == 0:
                round = i // 2
                ret += f"[Round {round}]{sep}{role}：{message}"
            else:
                ret += f"{sep}{role}："
                if message:
                    ret += message + sep
        return ret