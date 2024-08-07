from typing import Callable, Any, Optional

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.quantize import QConfig, QuantMethod
from haxllm.model.modules import RMSNorm, make_block_stack
from haxllm.model.parallel import GLUMlpBlock, DenseGeneral, Embed, SelfAttention
from haxllm.model.mixin import RematScanConfigMixin
from haxllm.chat.setting import register_chat_setting, ChatSetting


config_hub = {
    "llama-t": dict(
        hidden_size=1024,
        intermediate_size=2816,
        n_heads=8,
        n_layers=2,
    ),
    "llama-7b": dict(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
    ),
    "llama-13b": dict(
        hidden_size=5120,
        intermediate_size=13824,
        n_heads=40,
        n_layers=40,
    ),
    "llama-30b": dict(
        hidden_size=6656,
        intermediate_size=17920,
        n_heads=52,
        n_layers=60,
    ),
    "llama-65b": dict(
        hidden_size=8192,
        intermediate_size=22016,
        n_heads=64,
        n_layers=80,
    ),
    "llama2-7b": dict(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
        rms_norm_eps=1e-5,
        n_positions=4096,
    ),
    "llama2-13b": dict(
        hidden_size=5120,
        intermediate_size=13824,
        n_heads=40,
        n_layers=40,
        rms_norm_eps=1e-5,
        n_positions=4096,
    ),
    "yi-6b": dict(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=4,
        n_layers=32,
        rms_norm_eps=1e-5,
        n_positions=4096,
        vocab_size=64000,
        rope_theta=5000000.0,
    ),
    "yi-1.5-9b": dict(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=4,
        n_layers=48,
        rms_norm_eps=1e-6,
        n_positions=4096,
        vocab_size=64000,
        rope_theta=5000000.0,
    ),
    "yi-1.5-34b": dict(
        hidden_size=7168,
        intermediate_size=20480,
        n_heads=56,
        n_kv_heads=8,
        n_layers=60,
        rms_norm_eps=1e-6,
        n_positions=4096,
        vocab_size=64000,
        rope_theta=5000000.0,
    ),
    "llama3-8b": dict(
        hidden_size=4096,
        intermediate_size=14336,
        n_heads=32,
        n_kv_heads=8,
        n_layers=32,
        rms_norm_eps=1e-5,
        n_positions=8192,
        vocab_size=128256,
        rope_theta=500000.0,
        bos_token_id=128000,
        eos_token_id=128009,
    ),
}


@struct.dataclass
class TransformerConfig(RematScanConfigMixin):
    vocab_size: int = 32000
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 4096
    intermediate_size: int = 11008
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    n_layers: int = 32
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
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
    qconfig: Optional[QConfig] = None


def get_chat_setting(name=None):
    if name is None or name == 'none':
        return ChatSetting.none()
    elif name == VicunaChatSetting.name:
        return VicunaChatSetting()
    elif name == LLaMA2ChatSetting.name:
        return LLaMA2ChatSetting()
    else:
        raise ValueError(f"unknown chat config: {name}, must be one of {VicunaChatSetting.name}, {LLaMA2ChatSetting.name} or 'none'")


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
            qkv_bias=False,
            out_bias=False,
            decode=config.decode,
            rope=True,
            rope_theta=config.rope_theta,
            padding_left=config.padding_left,
            query_shard_axes=("X", "Y", None),
            kv_shard_axes=("X", "Y", None),
            kv_cache_shard_axes=(None, "X", "Y", None),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            shard_cache=config.shard_cache,
            qconfig=config.qconfig,
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
            qconfig=config.qconfig,
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
            shard_axes={"embedding": ("X", "Y")},
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
            shard_axes = {"kernel": ("Y", "X")}
        else:
            # shard output in training to avoid out of memory
            shard_axes = {'kernel': ("X", 'Y')}

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


def remap_state_dict(state_dict, head_dim=None, qconfig: Optional[QConfig] = None):
    q_method = qconfig.method if qconfig is not None else None
    n_layers = max([int(k.split('.')[2]) for k in state_dict.keys() if k.startswith("model.layers.")]) + 1
    hidden_size = state_dict['model.embed_tokens.weight'].shape[1]
    rope_key = 'model.layers.0.self_attn.rotary_emb.inv_freq'
    if rope_key in state_dict:
        head_dim = state_dict[rope_key].shape[0] * 2
    elif head_dim is None:
        head_dim = 128
    n_heads = hidden_size  // head_dim
    if q_method is None or q_method == QuantMethod.rtn_q8_0:
        sample_w = state_dict['model.layers.0.self_attn.k_proj.weight']
        n_kv_heads = sample_w.shape[0] // head_dim
        half_dtype = sample_w.dtype
    elif q_method in [QuantMethod.awq_q4, QuantMethod.gptq_q4]:
        q_key = 'model.layers.0.self_attn.q_proj.'
        qweight = state_dict[q_key + 'qweight']
        scales = state_dict[q_key + 'scales']
        half_dtype = scales.dtype
        # bits = scales.shape[-1] // qweight.shape[-1] // 2
        bits = 4
        assert qconfig.q_bits == bits
        bits_reduce = qconfig.w_bits // qconfig.q_bits
        if q_method == QuantMethod.awq_q4:
            group_size = qweight.shape[0] // scales.shape[0]
        else:
            group_size = qweight.shape[0] * bits_reduce // scales.shape[0]
        assert qconfig.group_size == group_size
        hidden_size_g = hidden_size // group_size
        n_kv_heads = state_dict['model.layers.0.self_attn.k_proj.scales'].shape[1] // head_dim
    assert half_dtype in [jnp.bfloat16, jnp.float16]

    root = {}
    root["wte"] = {"embedding": state_dict.pop("model.embed_tokens.weight").astype(half_dtype)}

    for d in range(n_layers):
        block_d = {}
        block_d["ln_1"] = {"scale": state_dict.pop(
            f"model.layers.{d}.input_layernorm.weight")}
        block_d["attn"] = {}
        block_d["mlp"] = {}
        for name in ["attn.query", "attn.key", "attn.value", "attn.out", "mlp.gate", "mlp.up", "mlp.down"]:
            dst_l, part = name.split('.')
            if dst_l == 'attn':
                src_l, src_l2 = "self_attn", part[0]
            else:
                src_l, src_l2 = "mlp", part
            prefix = f"model.layers.{d}.{src_l}.{src_l2}_proj"
            if q_method is None or q_method == QuantMethod.rtn_q8_0:
                kernel = state_dict.pop(f"{prefix}.weight").T
                quantize = q_method == QuantMethod.rtn_q8_0 and name in qconfig.q_layers
                if quantize:
                    kernel, scales = qconfig.quantize(kernel)
                if part == 'query':
                    kernel = kernel.reshape(hidden_size, n_heads, head_dim)
                elif part in ['key', 'value']:
                    kernel = kernel.reshape(hidden_size, n_kv_heads, head_dim)
                elif part == 'out':
                    kernel = kernel.reshape(n_heads, head_dim, hidden_size)
                params = {"kernel": kernel}
                if quantize:
                    params['scales'] = scales
            elif q_method in [QuantMethod.awq_q4, QuantMethod.gptq_q4]:
                qweight = state_dict.pop(f"{prefix}.qweight")
                qzeros = state_dict.pop(f"{prefix}.qzeros", None)
                scales = state_dict.pop(f"{prefix}.scales")
                qweight, qzeros = qconfig.requantize(qweight, qzeros)
                if qconfig.pack == 1:
                    div1, div2 = 1, 8
                elif qconfig.pack == 2:
                    div1, div2 = 2, 4
                elif qconfig.pack == 3:
                    div1, div2 = 8, 1
                if part == 'query':
                    qweight = qweight.reshape(hidden_size // div1, n_heads, head_dim // div2)
                    scales = scales.reshape(hidden_size_g, n_heads*head_dim)
                    if not qconfig.sym:
                        qzeros = qzeros.reshape(hidden_size_g, n_heads*head_dim)
                elif part in ['key', 'value']:
                    qweight = qweight.reshape(hidden_size // div1, n_kv_heads, head_dim // div2)
                    scales = scales.reshape(hidden_size_g, n_kv_heads*head_dim)
                    if not qconfig.sym:
                        qzeros = qzeros.reshape(hidden_size_g, n_kv_heads*head_dim)
                elif part == 'out':
                    # HACK
                    if qconfig.pack == 3:
                        qweight = qweight.reshape(n_heads, head_dim // div1, hidden_size // div2)
                    else:
                        qweight = qweight.reshape(n_heads // div1, head_dim, hidden_size // div2)
                    scales = scales.reshape(hidden_size_g, hidden_size)
                    if not qconfig.sym:
                        qzeros = qzeros.reshape(hidden_size_g, hidden_size)
                params = {"kernel": qweight, "scales": scales}
                if not qconfig.sym:
                    params['zeros'] = qzeros

            bias = state_dict.pop(f"{prefix}.bias", None)
            if bias is not None and not (bias == 0).all():
                if part == 'query':
                    bias = bias.reshape(n_heads, head_dim)
                elif part in ['key', 'value']:
                    bias = bias.reshape(n_kv_heads, head_dim)
                elif part == 'out':
                    bias = bias.reshape(hidden_size)
                params["bias"] = bias

            block_d[dst_l][part] = params
        block_d["ln_2"] = {"scale": state_dict.pop(
            f"model.layers.{d}.post_attention_layernorm.weight")}
        root[f"h_{d}"] = block_d

    root["ln_f"] = {"scale": state_dict.pop("model.norm.weight")}
    if "lm_head.weight" in state_dict:
        weight = state_dict.pop("lm_head.weight").T
    else:
        # tie_word_embeddings
        print("WARNING: use tied weights for lm_head")
        weight = root["wte"]["embedding"].T.copy()
    root["lm_head"] = {"kernel": weight.astype(half_dtype)}
    return root


@register_chat_setting()
class VicunaChatSetting:
    name = "vicuna_v1"
    system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    roles = ("USER", "ASSISTANT")
    stop_token_ids = (2,)

    def get_prompt(self, messages):
        seps = [" ", "</s>"]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret


@register_chat_setting()
class YiChatSetting:
    name = "Yi-chat"
    system = ""
    roles = ("user", "assistant")
    stop_token_ids = (2, 7,)

    def get_prompt(self, messages):
        bos = "<|im_start|>"
        eos = "<|im_end|>"
        ret = ""
        for i, (role, message) in enumerate(messages):
            ret += bos
            if message:
                ret += role + "\n" + message + eos + "\n"
            else:
                ret += role + "\n"
        return ret


@register_chat_setting()
class LLaMA3ChatSetting:
    name = "llama3-chat"
    system = "You are a helpful assistant."
    roles = ("user", "assistant")
    stop_token_ids = (128001, 128009,)

    def get_prompt(self, messages):
        boh = "<|start_header_id|>"
        eoh = "<|end_header_id|>"
        eos = "<|eot_id|>"
        ret = "<|begin_of_text|>"
        system = self.system
        if messages[0][0] == "system":
            system = messages[0][1]
            messages = messages[1:]
        system = system.strip()
        if system:
            ret += f"{boh}system{eoh}\n\n{system}{eos}"
        for i, (role, message) in enumerate(messages):
            ret += f"{boh}{role}{eoh}\n\n"
            if message:
                ret += message.strip() + eos
        return ret


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

@register_chat_setting()
class LLaMA2ChatSetting:
    name = "llama2-chat"
    system = DEFAULT_SYSTEM_PROMPT
    roles = ("user", "assistant")
    stop_token_ids = (2,)

    def get_prompt(self, messages):
        # TODO: add support for custom system prompt
        BOS = "<s>"
        EOS = "</s>"
        dialog = [
            {"role": role, "content": message}
            for role, message in messages
        ]
        if len(dialog) % 2 == 0:
            assert dialog[-1]["content"] is None
            dialog = dialog[:-1]
            append = True
        else:
            append = False
        if dialog[0]["role"] != "system":
            dialog = [
                {
                    "role": "system",
                    "content": DEFAULT_SYSTEM_PROMPT,
                }
            ] + dialog
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        ret = "".join(
            [
                f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ]
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        if append:
            ret += f"{BOS}{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
        return ret
