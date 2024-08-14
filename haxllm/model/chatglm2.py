from typing import Callable, Any

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.modules import RMSNorm, make_block_stack
from haxllm.model.parallel import GLUMlpBlock, DenseGeneral, Embed, SelfAttention
from haxllm.model.mixin import RematScanConfigMixin, RoPEScalingConfig
from haxllm.chat.setting import register_chat_setting


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


@struct.dataclass
class TransformerConfig(RematScanConfigMixin):
    vocab_size: int = 65024
    unpadded_vocab_size: int = 64794
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 4096
    intermediate_size: int = 13696
    n_heads: int = 32
    n_layers: int = 28
    num_groups: int = 2
    rms_norm_eps: float = 1e-5
    rope_scaling: RoPEScalingConfig = RoPEScalingConfig(rope_type="chatglm2")
    n_positions: int = 32768
    pad_token_id: int = 0
    eos_token_id: int = 2
    mask_token_id: int = 64789
    gmask_token_id: int = 64790
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    padding_left: bool = False
    memory_efficient_attention: bool = False
    decode: bool = False
    shard: bool = False
    shard_cache: bool = False


def get_chat_setting(name=None):
    if name is not None:
        assert name == ChatSetting.name
    return ChatSetting()


# TODO: skip apply_query_key_layer_scaling, same as query_key_layer_scaling_coeff in chatglm
# no difference in inference (forward), but may be in training (backward)
class TransformerBlock(nn.Module):
    config: TransformerConfig
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        inputs, padding_mask = inputs

        if config.memory_efficient_attention or config.decode:
            mask = None
        else:
            mask = nn.make_causal_mask(inputs[..., 0], dtype=jnp.bool_)  # (batch, 1, seq_len, seq_len)
            if padding_mask is not None:
                mask = mask & ~padding_mask[:, None, None, :]

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
            rope_scaling=config.rope_scaling,
            padding_left=config.padding_left,
            query_shard_axes=("X", "Y", None),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            shard_cache=config.shard_cache,
            name="attn")(x, mask, padding_mask)

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

        outputs = (y, padding_mask)

        if self.scan:
            return outputs, None
        else:
            return outputs


class TransformerModel(nn.Module):
    config: TransformerConfig
    block_cls: Callable = TransformerBlock

    @nn.compact
    def __call__(self, inputs, train):
        config = self.config
        remat = config.remat or config.remat_scan

        if not config.decode:
            assert inputs.shape[1] > 1, "input sequence length must be > 1 for training"

        embed_layer = Embed if remat else Embed
        x = embed_layer(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": (None, "Y")},
            shard=config.shard,
            name="wte")(inputs)

        padding_mask = None
        if config.padding_left and inputs.shape[1] > 1:
            padding_mask = jnp.equal(inputs, config.pad_token_id)

        x = make_block_stack(
            self.block_cls, config.n_layers, config)((x, padding_mask), train)[0]

        norm_layer = RMSNorm if remat else RMSNorm
        x = norm_layer(epsilon=config.rms_norm_eps,
                       dtype=config.dtype, name="ln_f")(x)
        return x


# class TransformerSequenceClassifier(nn.Module):
#     config: TransformerConfig

#     @nn.compact
#     def __call__(self, *, inputs, attn_mask, train=False):
#         config = self.config
#         x = TransformerModel(
#             config=config, name="transformer")(inputs, train)

#         batch_size = inputs.shape[0]
#         seq_len = jnp.not_equal(inputs, config.pad_token_id).sum(-1) - 1
#         x = x[jnp.arange(batch_size), seq_len]

#         x = DenseGeneral(
#             config.num_labels,
#             dtype=config.dtype,
#             kernel_init=config.kernel_init,
#             bias_init=config.bias_init,
#             name="score")(x)
#         return x


class TransformerLMHeadModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, input_ids, train=False):
        config = self.config
        x = TransformerModel(
            config=config, name="transformer")(inputs=input_ids, train=train)

        if config.decode:
            shard_axes = {"kernel": ("Y", None)}
        else:
            # shard output in training to avoid out of memory
            shard_axes = {'kernel': (None, 'Y')}

        x = DenseGeneral(
            config.vocab_size,
            use_bias=False,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            shard_axes=shard_axes,
            shard=config.shard,
            name="lm_head")(x)
        return x


def remap_state_dict(state_dict, head_dim=None):
    state_dict = {k.replace("transformer.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

    n_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith("layers.")]) + 1
    hidden_size = state_dict['embedding.word_embeddings.weight'].shape[1]
    # hard code for now
    if head_dim is None:
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


@register_chat_setting()
class ChatSetting:
    name = "chatglm2"
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
                round = i // 2 + 1  # only difference from CHAT_GLM
                ret += f"[Round {round}]{sep}{role}：{message}"
            else:
                ret += f"{sep}{role}："
                if message:
                    ret += message + sep
        return ret
