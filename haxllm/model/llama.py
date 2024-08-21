from typing import Callable, Any, Optional

from datetime import datetime

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.quantize import QConfig, QuantMethod
from haxllm.model.modules import RMSNorm, make_block_stack
from haxllm.model.parallel import GLUMlpBlock, DenseGeneral, Embed, SelfAttention
from haxllm.model.mixin import RematScanConfigMixin, RoPEScalingConfigMixin
from haxllm.chat.setting import register_chat_setting


def QwenConfig(**kwargs):
    base = dict(
        vocab_size=151936,
        qkv_bias=True,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151645,
        rope_theta=10000.0,
    )
    return {**base, **kwargs}


def Qwen2Config(**kwargs):
    base = dict(
        vocab_size=151936,
        qkv_bias=True,
        pad_token_id=151643,
        bos_token_id=151643,
        eos_token_id=151645,
        rope_theta=1000000.0,
    )
    return {**base, **kwargs}


def LlamaConfig(**kwargs):
    base = dict(
        vocab_size=32000,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
    )
    return {**base, **kwargs}


def Llama2Config(**kwargs):
    base = dict(
        vocab_size=32000,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )
    return {**base, **kwargs}


def InternLMConfig(**kwargs):
    base = dict(
        vocab_size=103168,
        qkv_bias=True,
        out_bias=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return {**base, **kwargs}


def ChatGLM2Config(**kwargs):
    base = dict(
        vocab_size=65024,
        qkv_bias=True,
        out_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rms_norm_eps=1e-5,
        rope_scaling=dict(
            rope_type="chatglm2",
        )
    )
    return {**base, **kwargs}


def YiConfig(**kwargs):
    base = dict(
        vocab_size=64000,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=5000000.0,
        rms_norm_eps=1e-5,
    )
    return {**base, **kwargs}


def Yi1_5Config(**kwargs):
    base = dict(
        vocab_size=64000,
        qkv_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=5000000.0,
        rms_norm_eps=1e-6,
    )
    return {**base, **kwargs}


def Llama3Config(**kwargs):
    base = dict(
        vocab_size=128256,
        qkv_bias=False,
        pad_token_id=0,
        bos_token_id=128000,
        eos_token_id=128009,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        rope_scaling=dict(
            rope_type="llama3",
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            max_position_embeddings=8192,
        ),
    )
    return {**base, **kwargs}


def MistralConfig(**kwargs):
    base = dict(
        vocab_size=32768,
        qkv_bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=1000000.0,
        rms_norm_eps=1e-5,
    )
    return {**base, **kwargs}


config_hub = {
    "qwen-7b": QwenConfig(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
    ),
    "qwen-14b": QwenConfig(
        hidden_size=5120,
        intermediate_size=13696,
        n_heads=40,
        n_layers=40,
    ),
    "qwen1.5-0.5b": Qwen2Config(
        hidden_size=1024,
        intermediate_size=2816,
        n_heads=16,
        n_layers=24,
    ),
    "qwen1.5-1.8b": Qwen2Config(
        hidden_size=2048,
        intermediate_size=5504,
        n_heads=16,
        n_layers=24,
    ),
    "qwen1.5-4b": Qwen2Config(
        hidden_size=2560,
        intermediate_size=6912,
        n_heads=20,
        n_layers=40,
    ),
    "qwen1.5-7b": Qwen2Config(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
    ),
    "qwen1.5-14b": Qwen2Config(
        hidden_size=5120,
        intermediate_size=13696,
        n_heads=40,
        n_layers=40,
        vocab_size=152064,
    ),
    "qwen1.5-32b": Qwen2Config(
        hidden_size=5120,
        intermediate_size=27392,
        n_heads=40,
        n_kv_heads=8,
        n_layers=64,
        vocab_size=152064,
    ),
    "qwen1.5-72b": Qwen2Config(
        hidden_size=8192,
        intermediate_size=24576,
        n_heads=64,
        n_kv_heads=64,
        n_layers=80,
        vocab_size=152064,
    ),
    "qwen2-1.5b": Qwen2Config(
        hidden_size=1536,
        intermediate_size=8960,
        n_heads=12,
        n_layers=28,
        n_kv_heads=2,
    ),
    "qwen2-7b": Qwen2Config(
        hidden_size=3584,
        intermediate_size=18944,
        n_heads=28,
        n_layers=28,
        n_kv_heads=4,
        vocab_size=152064,
    ),
    "llama-t": LlamaConfig(
        hidden_size=1024,
        intermediate_size=2816,
        n_heads=8,
        n_layers=2,
    ),
    "llama-7b": LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
    ),
    "llama-13b": LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        n_heads=40,
        n_layers=40,
    ),
    "llama-30b": LlamaConfig(
        hidden_size=6656,
        intermediate_size=17920,
        n_heads=52,
        n_layers=60,
    ),
    "llama-65b": LlamaConfig(
        hidden_size=8192,
        intermediate_size=22016,
        n_heads=64,
        n_layers=80,
    ),
    "llama2-7b": Llama2Config(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
    ),
    "llama2-13b": Llama2Config(
        hidden_size=5120,
        intermediate_size=13824,
        n_heads=40,
        n_layers=40,
    ),
    "internlm-7b": InternLMConfig(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
    ),
    "chatglm2-6b": ChatGLM2Config(
        hidden_size=4096,
        intermediate_size=13696,
        n_heads=32,
        n_layers=28,
        n_kv_heads=2,
    ),
    "yi-6b": YiConfig(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=4,
        n_layers=32,
    ),
    "yi-1.5-9b": Yi1_5Config(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=4,
        n_layers=48,
    ),
    "yi-1.5-34b": Yi1_5Config(
        hidden_size=7168,
        intermediate_size=20480,
        n_heads=56,
        n_kv_heads=8,
        n_layers=60,
    ),
    "llama3-8b": Llama3Config(
        hidden_size=4096,
        intermediate_size=14336,
        n_heads=32,
        n_kv_heads=8,
        n_layers=32,
    ),
    "mistral-7b-v0.3": MistralConfig(
        hidden_size=4096,
        intermediate_size=14336,
        n_heads=32,
        n_kv_heads=8,
        n_layers=32,
    ),
    "codestral-22b-v0.1": MistralConfig(
        hidden_size=6144,
        intermediate_size=16384,
        n_heads=48,
        n_kv_heads=8,
        n_layers=56,
    ),
}


@struct.dataclass
class TransformerConfig(RematScanConfigMixin, RoPEScalingConfigMixin):
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
    qkv_bias: bool = False
    out_bias: bool = False
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
            qkv_bias=config.qkv_bias,
            out_bias=config.out_bias,
            decode=config.decode,
            rope=True,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
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


def remap_qwen_state_dict(state_dict, head_dim=128):
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


def remap_llama_state_dict(state_dict, head_dim=None, qconfig: Optional[QConfig] = None):
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


def remap_chatglm2_state_dict(state_dict, head_dim=None):
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


REMAP_FN = {
    "llama": remap_llama_state_dict,
    "qwen": remap_qwen_state_dict,
    "chatglm": remap_chatglm2_state_dict,
}

def remap_state_dict(*args, **kwargs):
    if 'format' in kwargs:
        format = kwargs.pop('format')
    else:
        format = 'llama'
    return REMAP_FN[format](*args, **kwargs)


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
class LLaMA31InstructSetting:
    name = "llama31-instruct"
    system = ""
    roles = ("user", "assistant")
    stop_token_ids = (128001, 128009,)

    def get_prompt(self, messages):
        boh = "<|start_header_id|>"
        eoh = "<|end_header_id|>"
        eos = "<|eot_id|>"
        # ret = "<|begin_of_text|>"
        ret = ""
        system = self.system
        if messages[0][0] == "system":
            system = messages[0][1]
            messages = messages[1:]
        system = system.strip()
        if system:
            ret += f"{boh}system{eoh}\n\n" \
                   f"Cutting Knowledge Date: December 2023\n" \
                   f"Today Date: {datetime.now().strftime('%d %b %Y')}\n\n" \
                   f"{system}{eos}"
        for i, (role, message) in enumerate(messages):
            ret += f"{boh}{role}{eoh}\n\n"
            if message:
                ret += message.strip() + eos
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
        # ret = "<|begin_of_text|>"
        ret = ""
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


@register_chat_setting()
class MistralSetting:
    name = "mistral-instruct"
    system = ""
    roles = ("user", "assistant")
    stop_token_ids = (2,)

    def get_prompt(self, messages):
        B_INST, E_INST = "[INST]", "[/INST]"
        eos = "</s>"
        # ret = "<s>"
        ret = ""
        system = self.system
        if messages[0][0] == "system":
            system = messages[0][1]
            messages = messages[1:]
        system = system.strip()
        n = len(messages)
        for i, (role, content) in enumerate(messages):
            if role == self.roles[0]:
                if i % 2 != 0:
                    raise ValueError(f"After the optional system message, conversation roles must alternate user/assistant/user/assistant/...")
                if system and i in [n - 1, n - 2]:
                    ret += f"{B_INST} {system}\n\n{content}{E_INST}"
                else:
                    ret += f"{B_INST} {content}{E_INST}"
            else:
                if content:
                    ret += f" {content}{eos}"
                else:
                    ret += f" "
        return ret


@register_chat_setting()
class InternLMChatSetting:
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


@register_chat_setting()
class ChatGLM2Setting:
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
