from typing import Callable, Any

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.modules import RMSNorm, make_block_stack
from haxllm.model.parallel import GLUMlpBlock, DenseGeneral, Embed, SelfAttention
from haxllm.model.mixin import RematScanConfigMixin
from haxllm.chat.setting import register_chat_setting


config_hub = {
    "qwen-7b": dict(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
    ),
    "qwen-14b": dict(
        vocab_size=152064,
        hidden_size=5120,
        intermediate_size=13696,
        n_heads=40,
        n_layers=40,
    ),
}


@struct.dataclass
class TransformerConfig(RematScanConfigMixin):
    vocab_size: int = 151936
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 4096
    intermediate_size: int = 11008
    n_heads: int = 32
    n_layers: int = 32
    rms_norm_eps: float = 1e-6
    n_positions: int = 8192
    pad_token_id: int = 151850  # last id of unpadded vocab, unpossible to be used (<extra_199>)
    bos_token_id: int = 151643
    eos_token_id: int = 151643
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
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            qkv_bias=True,
            out_bias=False,
            decode=config.decode,
            rope=True,
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


def remap_state_dict(state_dict, head_dim=128):
    n_layers = max([int(k.split('.')[2]) for k in state_dict.keys() if k.startswith("transformer.h.")]) + 1
    hidden_size = state_dict['transformer.wte.weight'].shape[1]
    if head_dim is None:
        head_dim = 128
    n_heads = hidden_size  // head_dim

    root = {}
    root["wte"] = {"embedding": state_dict.pop("transformer.wte.weight")}

    for d in range(n_layers):
        block_d = {}
        block_d["ln_1"] = {"scale": state_dict.pop(
            f"transformer.h.{d}.ln_1.weight")}
        c_attn_weight = state_dict[f"transformer.h.{d}.attn.c_attn.weight"].T
        c_attn_bias = state_dict[f"transformer.h.{d}.attn.c_attn.bias"]
        block_d["attn"] = {
            "query": {
                "kernel": c_attn_weight[:, 0:hidden_size].reshape(hidden_size, n_heads, head_dim),
                "bias": c_attn_bias[0:hidden_size].reshape(n_heads, head_dim),
            },
            "key": {
                "kernel": c_attn_weight[:, hidden_size:hidden_size*2].reshape(hidden_size, n_heads, head_dim),
                "bias": c_attn_bias[hidden_size:hidden_size*2].reshape(n_heads, head_dim),
            },
            "value": {
                "kernel": c_attn_weight[:, hidden_size*2:hidden_size*3].reshape(hidden_size, n_heads, head_dim),
                "bias": c_attn_bias[2*hidden_size:hidden_size*3].reshape(n_heads, head_dim),
            },
            "out": {
                "kernel": state_dict.pop(f"transformer.h.{d}.attn.c_proj.weight").T.reshape(n_heads, head_dim, hidden_size),
            },
        }
        block_d["ln_2"] = {"scale": state_dict.pop(f"transformer.h.{d}.ln_2.weight")}
        block_d["mlp"] = {
            "gate": {"kernel": state_dict.pop(f"transformer.h.{d}.mlp.w2.weight").T},
            "up": {"kernel": state_dict.pop(f"transformer.h.{d}.mlp.w1.weight").T},
            "down": {"kernel": state_dict.pop(f"transformer.h.{d}.mlp.c_proj.weight").T},
        }
        root[f"h_{d}"] = block_d

    root["ln_f"] = {"scale": state_dict.pop("transformer.ln_f.weight")}
    root["lm_head"] = {"kernel": state_dict.pop("lm_head.weight").T}
    return root


@register_chat_setting()
class ChatSetting:
    name = "qwen"
    system = "You are a helpful assistant."
    roles = ("user", "assistant")
    stop_token_ids = (151643,)

    def get_prompt(self, messages):
        return encode_message(messages, self.system)


def encode_message(messages, system):
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    if messages[0][0] == "system":
        system = messages[0][1]
        messages = messages[1:]
    system = system.strip()

    sep = "\n"
    ret = f"{im_start}system{sep}{system}{im_end}"

    for i, (role, message) in enumerate(messages):
        if i % 2 == 0:
            ret += f"{sep}{im_start}{role}{sep}{message}{im_end}"
        else:
            ret += f"{sep}{im_start}{role}{sep}"
            if message:
                ret += f"{message}{im_end}"
    return ret