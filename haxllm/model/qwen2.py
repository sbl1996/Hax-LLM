from typing import Callable, Any, Optional

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.modules import RMSNorm, make_block_stack
from haxllm.model.parallel import GLUMlpBlock, DenseGeneral, Embed, SelfAttention
from haxllm.model.mixin import RematScanConfigMixin
from haxllm.chat.setting import register_chat_setting
from haxllm.model.llama import remap_state_dict as llama_remap_state_dict
from haxllm.model.qwen import encode_message


config_hub = {
    "qwen1.5-0.5b": dict(
        hidden_size=1024,
        intermediate_size=2816,
        n_heads=16,
        n_layers=24,
    ),
    "qwen1.5-1.8b": dict(
        hidden_size=2048,
        intermediate_size=5504,
        n_heads=16,
        n_layers=24,
    ),
    "qwen1.5-4b": dict(
        hidden_size=2560,
        intermediate_size=6912,
        n_heads=20,
        n_layers=40,
    ),
    "qwen1.5-7b": dict(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
    ),
    "qwen1.5-14b": dict(
        hidden_size=5120,
        intermediate_size=13696,
        n_heads=40,
        n_layers=40,
        vocab_size=152064,
    ),
    "qwen1.5-32b": dict(
        hidden_size=5120,
        intermediate_size=27392,
        n_heads=40,
        n_kv_heads=8,
        n_layers=64,
        vocab_size=152064,
    ),
    "qwen1.5-72b": dict(
        hidden_size=8192,
        intermediate_size=24576,
        n_heads=64,
        n_kv_heads=64,
        n_layers=80,
        vocab_size=152064,
    ),
    "qwen2-1.5b": dict(
        hidden_size=1536,
        intermediate_size=8960,
        n_heads=12,
        n_layers=28,
        n_kv_heads=2,
    ),
    "qwen2-7b": dict(
        hidden_size=3584,
        intermediate_size=18944,
        n_heads=28,
        n_layers=28,
        n_kv_heads=4,
        vocab_size=152064,
    ),
}


@struct.dataclass
class TransformerConfig(RematScanConfigMixin):
    vocab_size: int = 151936
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 1024
    intermediate_size: int = 2816
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    n_layers: int = 24
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    n_positions: int = 32768
    pad_token_id: int = 151643
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    padding_left: bool = False
    memory_efficient_attention: bool = False
    decode: bool = False
    shard: bool = False
    shard_cache: bool = False
    quantize: bool = False


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
            multi_query_groups=config.n_kv_heads,
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            qkv_bias=True,
            out_bias=False,
            decode=config.decode,
            rope=True,
            rope_theta=config.rope_theta,
            padding_left=config.padding_left,
            query_shard_axes=("X", "Y", None),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            shard_cache=config.shard_cache,
            quantize=config.quantize,
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
            quantize=config.quantize,
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


remap_state_dict = llama_remap_state_dict



@register_chat_setting()
class ChatSetting:
    name = "qwen2"
    system = "You are a helpful assistant"
    roles = ("user", "assistant")
    stop_token_ids = (151643,)

    def get_prompt(self, messages):
        return encode_message(messages, self.system)
