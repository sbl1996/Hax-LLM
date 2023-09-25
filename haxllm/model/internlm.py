from typing import Callable, Any
import dataclasses

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.modules import RMSNorm, make_block_stack
from haxllm.model.parallel import GLUMlpBlock, DenseGeneral, Embed, SelfAttention
from haxllm.model.mixin import RematScanConfigMixin
from haxllm.chat.setting import register_chat_setting


config_hub = {
    "internlm-7b": dict(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
    ),
}


@struct.dataclass
class TransformerConfig(RematScanConfigMixin):
    vocab_size: int = 103168
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 4096
    intermediate_size: int = 11008
    n_heads: int = 32
    n_layers: int = 32
    rms_norm_eps: float = 1e-6
    n_positions: int = 2048
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
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
            out_bias=True,
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


def remap_state_dict(state_dict):
    n_layers = max([int(k.split('.')[2]) for k in state_dict.keys() if k.startswith("model.layers.")]) + 1
    hidden_size = state_dict['model.embed_tokens.weight'].shape[1]
    head_dim = state_dict['model.layers.0.self_attn.rotary_emb.inv_freq'].shape[0] * 2
    n_heads = hidden_size  // head_dim

    root = {}
    root["wte"] = {"embedding": state_dict.pop("model.embed_tokens.weight")}

    for d in range(n_layers):
        block_d = {}
        block_d["ln_1"] = {"scale": state_dict.pop(
            f"model.layers.{d}.input_layernorm.weight")}
        block_d["attn"] = {
            "query": {
                "kernel": state_dict.pop(f"model.layers.{d}.self_attn.q_proj.weight").T.reshape(hidden_size, n_heads, head_dim),
                "bias": state_dict.pop(f"model.layers.{d}.self_attn.q_proj.bias").reshape(n_heads, head_dim),
            },
            "key": {
                "kernel": state_dict.pop(f"model.layers.{d}.self_attn.k_proj.weight").T.reshape(hidden_size, n_heads, head_dim),
                "bias": state_dict.pop(f"model.layers.{d}.self_attn.k_proj.bias").reshape(n_heads, head_dim),
            },
            "value": {
                "kernel": state_dict.pop(f"model.layers.{d}.self_attn.v_proj.weight").T.reshape(hidden_size, n_heads, head_dim),
                "bias": state_dict.pop(f"model.layers.{d}.self_attn.v_proj.bias").reshape(n_heads, head_dim),
            },
            "out": {
                "kernel": state_dict.pop(f"model.layers.{d}.self_attn.o_proj.weight").T.reshape(n_heads, head_dim, hidden_size),
                "bias": state_dict.pop(f"model.layers.{d}.self_attn.o_proj.bias"),
            },
        }
        block_d["ln_2"] = {"scale": state_dict.pop(
            f"model.layers.{d}.post_attention_layernorm.weight")}
        block_d["mlp"] = {
            "gate": {"kernel": state_dict.pop(f"model.layers.{d}.mlp.gate_proj.weight").T},
            "up": {"kernel": state_dict.pop(f"model.layers.{d}.mlp.up_proj.weight").T},
            "down": {"kernel": state_dict.pop(f"model.layers.{d}.mlp.down_proj.weight").T},
        }
        root[f"h_{d}"] = block_d

    root["ln_f"] = {"scale": state_dict.pop("model.norm.weight")}
    root["lm_head"] = {"kernel": state_dict.pop("lm_head.weight").T}
    return root


@register_chat_setting()
class ChatSetting:
    name = "internlm"
    system = ""
    roles = ("<|User|>", "<|Bot|>")
    stop_token_ids = (2, 103028)

    def get_prompt(self, messages):
        r'''
        prompt = ""
        for record in history:
            prompt += f"""<s><|User|>:{record[0]}<eoh>\n<|Bot|>:{record[1]}<eoa>\n"""
        if len(prompt) == 0:
            prompt += "<s>"
        prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""            
        '''
        sep = "<eoh>\n"
        sep2 = "<eoa>\n"
        ret = ""
        m = messages
        n = len(m)
        n = n - 2 if n % 2 == 0 else n - 1
        for i in range(0, n, 2):
            role1, message1 = m[i]
            role2, message2 = m[i+1]
            ret += f"""<s>{role1}:{message1}{sep}{role2}:{message2}{sep2}"""
        if len(ret) == 0:
            assert len(m) - n == 2
            role1, message1 = m[-2]
            role2, message2 = m[-1]
            ret += f"""<s>{role1}:{message1}{sep}{role2}:"""
            if message2:
                ret += message2 + sep2
        else:
            if len(m) - n == 1:
                role, message = m[-1]
                ret += f"""<s>{role}:{message}{sep2}"""
            else:
                role1, message1 = m[-2]
                role2, message2 = m[-1]
                ret += f"""<s>{role1}:{message1}{sep}{role2}:"""
                if message2:
                    ret += message2 + sep2
        return ret