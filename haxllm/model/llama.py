from typing import Callable, Any, Optional

import numpy as np
import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.quantize import QConfig, QuantMethod, QuantSource
from haxllm.model.modules import RMSNorm, make_block_stack
from haxllm.model.parallel import GLUMlpBlock, DenseGeneral, Embed, SelfAttention, ShardModule
from haxllm.model.mixin import RematScanConfigMixin, RoPEScalingConfigMixin
from haxllm.model.lora import LoraConfig


def QwenConfig(**kwargs):
    base = dict(
        vocab_size=151936,
        qkv_bias=True,
        pad_token_id=151643,  # although same as bos_token_id, but actually not used
        bos_token_id=151643,
        eos_token_id=151645,
        rope_theta=10000.0,
        n_positions=32768,
    )
    return {**base, **kwargs}


def LlamaConfig(**kwargs):
    base = dict(
        vocab_size=32000,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        n_positions=4096,
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
        n_positions=4096,
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
        n_positions=2048,
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
        n_positions=32768,
        rope_scaling=dict(
            rope_type="chatglm",
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
        n_positions=4096,
    )
    return {**base, **kwargs}


def Yi1_5Config(**kwargs):
    base = dict(
        vocab_size=64000,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=5000000.0,
        rms_norm_eps=1e-6,
        n_positions=4096,
    )
    return {**base, **kwargs}


def Llama3Config(**kwargs):
    base = dict(
        vocab_size=128256,
        pad_token_id=0,
        bos_token_id=128000,
        eos_token_id=128009,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        n_positions=8192,
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
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=1000000.0,
        rms_norm_eps=1e-5,
        n_positions=32768,
    )
    return {**base, **kwargs}


def MistralNemoConfig(**kwargs):
    base = dict(
        vocab_size=131072,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=1000000.0,
        rms_norm_eps=1e-5,
        n_positions=32768,
        head_dim=128,
    )
    return {**base, **kwargs}


def Qwen2Config(**kwargs):
    base = dict(
        vocab_size=151936,
        qkv_bias=True,
        pad_token_id=151643,  # although same as bos_token_id, but actually not used
        bos_token_id=151643,
        eos_token_id=151645,
        rope_theta=1000000.0,
        n_positions=32768,
    )
    return {**base, **kwargs}


def InternLM2_5Config(**kwargs):
    base = dict(
        vocab_size=92544,
        pad_token_id=0,  # <unk>, unpossible to be used
        bos_token_id=1,
        eos_token_id=2,
        rms_norm_eps=1e-5,
        n_positions=32768,
        rope_theta=1000000.0,
    )
    return {**base, **kwargs}


def Phi3Config(**kwargs):
    base = dict(
        vocab_size=32064,
        pad_token_id=0,  # <unk>, unpossible to be used
        bos_token_id=1,
        eos_token_id=32000,
        rms_norm_eps=1e-5,
        n_positions=32768,
        rope_theta=10000.0,
        rope_scaling=dict(
            rope_type='longrope',
            max_position_embeddings=4096,
            long_factor=[1.0800000429153442,1.1100000143051147,1.1399999856948853,1.340000033378601,1.5899999141693115,1.600000023841858,1.6200000047683716,2.620000123977661,3.2300000190734863,3.2300000190734863,4.789999961853027,7.400000095367432,7.700000286102295,9.09000015258789,12.199999809265137,17.670000076293945,24.46000099182129,28.57000160217285,30.420001983642578,30.840002059936523,32.590003967285156,32.93000411987305,42.320003509521484,44.96000289916992,50.340003967285156,50.45000457763672,57.55000305175781,57.93000411987305,58.21000289916992,60.1400032043457,62.61000442504883,62.62000274658203,62.71000289916992,63.1400032043457,63.1400032043457,63.77000427246094,63.93000411987305,63.96000289916992,63.970001220703125,64.02999877929688,64.06999969482422,64.08000183105469,64.12000274658203,64.41000366210938,64.4800033569336,64.51000213623047,64.52999877929688, 64.83999633789062],
            short_factor=[1.0,1.0199999809265137,1.0299999713897705,1.0299999713897705,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0699999332427979,1.0999999046325684,1.1099998950958252,1.1599998474121094,1.1599998474121094,1.1699998378753662,1.2899998426437378,1.339999794960022,1.679999828338623,1.7899998426437378,1.8199998140335083,1.8499997854232788,1.8799997568130493,1.9099997282028198,1.9399996995925903,1.9899996519088745,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0799996852874756,2.0899996757507324,2.189999580383301,2.2199995517730713,2.5899994373321533,2.729999542236328,2.749999523162842,2.8399994373321533],
        )
    )
    return {**base, **kwargs}


def GLM4Config(**kwargs):
    base = dict(
        vocab_size=151552,
        qkv_bias=True,
        out_bias=False,
        pad_token_id=151330,  # [MASK], unpossible to be used
        bos_token_id=151329,
        eos_token_id=151329,
        rms_norm_eps=0.00000015625,
        n_positions=131072,
        rope_theta=5000000.0,
        rope_scaling=dict(
            rope_type="chatglm",
        )
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
        tie_word_embeddings=True,
    ),
    "qwen2-7b": Qwen2Config(
        hidden_size=3584,
        intermediate_size=18944,
        n_heads=28,
        n_layers=28,
        n_kv_heads=4,
        vocab_size=152064,
    ),
    "qwen2-72b": Qwen2Config(
        hidden_size=8192,
        intermediate_size=29696,
        n_heads=64,
        n_layers=80,
        n_kv_heads=8,
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
    "mistral-nemo-12b": MistralNemoConfig(
        hidden_size=5120,
        intermediate_size=14336,
        n_heads=32,
        n_kv_heads=8,
        n_layers=40,
    ),
    "internlm2.5-1.8b": InternLM2_5Config(
        hidden_size=2048,
        intermediate_size=8192,
        n_heads=16,
        n_kv_heads=8,
        n_layers=24,
        rope_scaling=dict(
            factor=2.0,
            rope_type='dynamic',
            max_position_embeddings=32768,
        )
    ),
    "internlm2.5-7b": InternLM2_5Config(
        hidden_size=4096,
        intermediate_size=14336,
        n_heads=32,
        n_kv_heads=8,
        n_layers=32,
        rope_scaling=dict(
            factor=2.0,
            rope_type='dynamic',
            max_position_embeddings=32768,
        )
    ),
    "internlm2.5-20b": InternLM2_5Config(
        hidden_size=6144,
        intermediate_size=16384,
        n_heads=48,
        n_kv_heads=8,
        n_layers=48,
        rope_theta=50000000.0,
        rope_scaling=dict(
            factor=2.5,
            rope_type='dynamic',
            max_position_embeddings=32768,
        )
    ),
    "phi-3.5-mini-instruct": Phi3Config(
        hidden_size=3072,
        intermediate_size=8192,
        n_heads=32,
        n_layers=32,
    ),
    "glm4-9b": GLM4Config(
        hidden_size=4096,
        intermediate_size=13696,
        n_heads=32,
        n_layers=40,
        n_kv_heads=2,
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
    head_dim: Optional[int] = None
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
    decode: bool = False
    shard: bool = False
    shard_cache: bool = False
    qconfig: Optional[QConfig] = None
    tie_word_embeddings: bool = False
    lora_config: Optional[LoraConfig] = None

    def get_dense_cls(self, name):
        if self.lora_config is None:
            return DenseGeneral
        config = self.lora_config
        assert len(config.attn_lora_r) == 4
        assert len(config.mlp_lora_r) == 3
        return config.create_dense_cls(name)


class TransformerBlock(nn.Module):
    config: TransformerConfig
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        # resid_pre
        inputs, padding_mask = inputs

        self.sow('intermediates', 'resid_pre', inputs)

        x = RMSNorm(epsilon=config.rms_norm_eps,
                    dtype=config.dtype, name="ln_1")(inputs)
        # attn_out
        x = SelfAttention(
            num_heads=config.n_heads,
            num_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            qkv_bias=config.qkv_bias,
            out_bias=config.out_bias,
            is_causal=True,
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
            dense_cls=config.get_dense_cls("attn"),
            name="attn")(x, padding_mask=padding_mask)

        # mlp_in = resid_mid = attn_out + resid_pre
        x = x + inputs

        y = RMSNorm(epsilon=config.rms_norm_eps,
                    dtype=config.dtype, name="ln_2")(x)
        # mlp_out
        y = GLUMlpBlock(
            intermediate_size=config.intermediate_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            use_bias=False,
            shard_axes1=("X", "Y"),
            shard_axes2=("Y", "X"),
            shard=config.shard,
            qconfig=config.qconfig,
            dense_cls=config.get_dense_cls("mlp"),
            name="mlp")(y)

        # resid_post = resid_mid + mlp_out
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
        x, padding_mask = inputs
        x = make_block_stack(
            self.block_cls, config.n_layers, config)((x, padding_mask), train)[0]

        norm_layer = RMSNorm
        x = norm_layer(epsilon=config.rms_norm_eps,
                       dtype=config.dtype, name="ln_f")(x)
        return x


class TransformerLMHeadModel(ShardModule):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, input_ids, train=False):
        config = self.config

        if not config.decode:
            assert input_ids.shape[1] > 1, "input sequence length must be > 1 for training"

        embed_layer = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": ("X", "Y")},
            shard=config.shard,
            name="wte")
        x = embed_layer(input_ids)

        padding_mask = None
        if config.decode and config.padding_left and input_ids.shape[1] > 1:
            padding_mask = jnp.equal(input_ids, config.pad_token_id)

        x = TransformerModel(
            config=config, name="transformer")(inputs=(x, padding_mask), train=train)

        if config.tie_word_embeddings:
            x = embed_layer.attend(x)
        else:
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


def remap_llama_state_dict(state_dict, head_dim=None):
    n_layers = max([int(k.split('.')[2]) for k in state_dict.keys() if k.startswith("model.layers.")]) + 1
    hidden_size = state_dict['model.embed_tokens.weight'].shape[1]
    rope_key = 'model.layers.0.self_attn.rotary_emb.inv_freq'
    if rope_key in state_dict:
        head_dim = state_dict[rope_key].shape[0] * 2
    elif head_dim is None:
        head_dim = 128
    # n_heads = hidden_size  // head_dim
    sample_kv_key = 'model.layers.0.self_attn.k_proj.weight'
    if sample_kv_key in state_dict:
        sample_kv = state_dict['model.layers.0.self_attn.k_proj.weight']
        n_kv_heads = sample_kv.shape[0] // head_dim
        half_dtype = sample_kv.dtype
    else:
        scales = state_dict['model.layers.0.self_attn.k_proj.scales']
        n_kv_heads = scales.shape[1] // head_dim
        half_dtype = scales.dtype
        if half_dtype == jnp.float32:
            half_dtype = jnp.bfloat16

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
            weight_key = f"{prefix}.weight"
            if weight_key in state_dict:
                kernel = state_dict.pop(f"{prefix}.weight").T
                if part == 'query':
                    kernel = kernel.reshape(hidden_size, -1, head_dim)
                elif part in ['key', 'value']:
                    kernel = kernel.reshape(hidden_size, n_kv_heads, head_dim)
                elif part == 'out':
                    kernel = kernel.reshape(-1, head_dim, hidden_size)
                params = {"kernel": kernel}
            else:
                qweight = state_dict.pop(f"{prefix}.qweight")
                qzeros = state_dict.pop(f"{prefix}.qzeros", None)
                scales = state_dict.pop(f"{prefix}.scales")
                g_idx = state_dict.pop(f"{prefix}.g_idx", None)
                params = {"qweight": qweight, "qzeros": qzeros, "scales": scales}
                if g_idx is not None:
                    params["g_idx"] = g_idx
            bias = state_dict.pop(f"{prefix}.bias", None)
            if bias is not None and not (bias == 0).all():
                if part == 'query':
                    bias = bias.reshape(-1, head_dim)
                elif part in ['key', 'value']:
                    bias = bias.reshape(n_kv_heads, head_dim)
                elif part == 'out':
                    bias = bias.reshape(hidden_size)
                params["bias"] = bias

            block_d[dst_l][part] = params
        block_d["ln_2"] = {"scale": state_dict.pop(
            f"model.layers.{d}.post_attention_layernorm.weight")}

        # gemma2
        ln_2p = state_dict.pop(f"model.layers.{d}.post_feedforward_layernorm.weight", None)
        if ln_2p is not None:
            block_d["ln_1p"] = {"scale": block_d["ln_2"]["scale"]}
            block_d["ln_2"]['scale'] = state_dict.pop(
                f"model.layers.{d}.pre_feedforward_layernorm.weight")
            block_d["ln_2p"] = {"scale": ln_2p}

        root[f"h_{d}"] = block_d

    root["ln_f"] = {"scale": state_dict.pop("model.norm.weight")}
    if "lm_head.weight" in state_dict:
        weight = state_dict.pop("lm_head.weight").T
        root["lm_head"] = {"kernel": weight.astype(half_dtype)}
    else:
        # tie_word_embeddings
        print("WARNING: use tied weights for lm_head")
    return root


def quantize_llama_to_q8(root, qconfig: QConfig):
    # half -> int8
    n_layers = len([k for k in root.keys() if k.startswith("h_")])
    q_method = qconfig.method
    if qconfig.source != QuantSource.half or q_method != QuantMethod.rtn_q8_0:
        raise NotImplementedError("only support rtn_q8_0")

    for i in range(n_layers):
        d = root[f"h_{i}"]
        for name in ["attn.query", "attn.key", "attn.value", "attn.out", "mlp.gate", "mlp.up", "mlp.down"]:
            if name not in qconfig.q_layers:
                continue
            part1, part2 = name.split('.')
            params = d[part1][part2]
            kernel = params['kernel']
            shape = kernel.shape
            # reshape to ensure scales are always 2-dim
            if part2 in ['query', 'key', 'value']:
                kernel = kernel.reshape(shape[0], -1)
            elif part2 == 'out':
                # h, d -> h*d
                kernel = kernel.reshape(-1, *shape[2:])
            kernel, scales = qconfig.quantize(kernel)
            kernel = kernel.reshape(shape)
            params['kernel'] = kernel
            params['scales'] = scales
    return root


def convert_llama_q_params(root, qconfig: QConfig, head_dim=128):
    # TODO: hard code head_dim
    # TODO: decompose remap and quantize
    n_layers = len([k for k in root.keys() if k.startswith("h_")])
    q_source = qconfig.source
    if q_source not in [QuantSource.autoawq_q4, QuantSource.autogptq_q4, QuantSource.autogptq_q8]:
        raise NotImplementedError(f"only support awq_q4 and gptq_q4, but got {q_source}")

    hidden_size = root['wte']['embedding'].shape[1]
    n_heads = hidden_size  // head_dim
    q_params = root['h_0']['attn']['query']
    qweight = q_params['qweight']
    scales = q_params['scales']
    if q_source == QuantSource.autogptq_q8:
        group_size = qweight.shape[0] * 4 // scales.shape[0]
    else:
        assert qconfig.w_bits == 4
        bits_reduce = qconfig.f_bits // qconfig.w_bits
        if q_source == QuantSource.autoawq_q4:
            group_size = qweight.shape[0] // scales.shape[0]
        else:
            group_size = qweight.shape[0] * bits_reduce // scales.shape[0]
    assert qconfig.group_size == group_size, f"group size mismatch: {qconfig.group_size} vs {group_size}"
    hidden_size_g = hidden_size // group_size
    n_kv_heads = root['h_0']['attn']['key']['scales'].shape[1] // head_dim

    skip_g_idx = None
    for i in range(n_layers):
        d = root[f"h_{i}"]
        for name in ["attn.query", "attn.key", "attn.value", "attn.out", "mlp.gate", "mlp.up", "mlp.down"]:
            part1, part2 = name.split('.')
            params = d[part1][part2]
            qweight = params.pop('qweight')
            qzeros = params.pop('qzeros')
            scales = params.pop('scales')
            g_idx = params.pop('g_idx', None)
            if g_idx is not None:
                if skip_g_idx is None:
                    n_groups = g_idx.shape[0] // group_size
                    skip_g_idx = bool((g_idx.reshape(-1, group_size) == np.arange(n_groups)[:, None]).all())
                    if skip_g_idx is True:
                        print("Skip g_idx")
                elif skip_g_idx is True:
                    g_idx = None
                else:
                    assert qconfig.use_g_idx
            qweight, qzeros = qconfig.requantize(qweight, qzeros)
            if q_source == QuantSource.autogptq_q8:
                div1, div2 = 1, 1
            elif q_source in [QuantSource.autoawq_q4, QuantSource.autogptq_q4]:
                div1, div2 = 2, 4
            if part2 == 'query':
                qweight = qweight.reshape(hidden_size // div1, n_heads, head_dim // div2)
                scales = scales.reshape(hidden_size_g, n_heads*head_dim)
                if not qconfig.sym:
                    qzeros = qzeros.reshape(hidden_size_g, n_heads*head_dim)
            elif part2 in ['key', 'value']:
                qweight = qweight.reshape(hidden_size // div1, n_kv_heads, head_dim // div2)
                scales = scales.reshape(hidden_size_g, n_kv_heads*head_dim)
                if not qconfig.sym:
                    qzeros = qzeros.reshape(hidden_size_g, n_kv_heads*head_dim)
            elif part2 == 'out':
                qweight = qweight.reshape(n_heads // div1, head_dim, hidden_size // div2)
                scales = scales.reshape(hidden_size_g, hidden_size)
                if not qconfig.sym:
                    qzeros = qzeros.reshape(hidden_size_g, hidden_size)
            params['kernel'] = qweight
            params['scales'] = scales
            if not qconfig.sym:
                params['zeros'] = qzeros
            if g_idx is not None:
                params['g_idx'] = g_idx
    return root


def remap_chatglm_state_dict(state_dict, head_dim=None):
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
        c_attn_weight = c_attn_weight.reshape(hidden_size, (n_heads + 2 * num_groups) * head_dim)
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


def remap_internlm2_state_dict(state_dict, head_dim=None):
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    n_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith("layers.")]) + 1
    hidden_size = state_dict['tok_embeddings.weight'].shape[1]
    if head_dim is None:
        head_dim = 128
    n_heads = hidden_size  // head_dim
    n_kv_heads = (state_dict['layers.0.attention.wqkv.weight'].shape[0] // head_dim - n_heads) // 2
    g = n_heads // n_kv_heads

    root = {}
    root["wte"] = {"embedding": state_dict.pop("tok_embeddings.weight")}

    # TransformerBlock
    for d in range(n_layers):
        block_d = {}
        block_d["ln_1"] = {
            "scale": state_dict.pop(f"layers.{d}.attention_norm.weight"),
        }
        w_qkv = state_dict[f"layers.{d}.attention.wqkv.weight"].T
        w_qkv = w_qkv.reshape(hidden_size, n_kv_heads, g + 2, head_dim)

        block_d["attn"] = {
            "query": {
                "kernel": w_qkv[:, :, :g].reshape(hidden_size, n_heads, head_dim)
            },
            "key": {
                "kernel": w_qkv[:, :, -2].reshape(hidden_size, n_kv_heads, head_dim)
            },
            "value": {
                "kernel": w_qkv[:, :, -1].reshape(hidden_size, n_kv_heads, head_dim)
            },
            "out": {
                "kernel": state_dict.pop(
                    f"layers.{d}.attention.wo.weight").T.reshape(n_heads, head_dim, hidden_size),
            },
        }
        block_d["ln_2"] = {
            "scale": state_dict.pop(f"layers.{d}.ffn_norm.weight"),
        }

        block_d["mlp"] = {
            "gate": {"kernel": state_dict.pop(f"layers.{d}.feed_forward.w1.weight").T},
            "up": {"kernel": state_dict.pop(f"layers.{d}.feed_forward.w3.weight").T},
            "down": {"kernel": state_dict.pop(f"layers.{d}.feed_forward.w2.weight").T},
        }
        root[f"h_{d}"] = block_d

    root["ln_f"] = {
        "scale": state_dict.pop("norm.weight"),
    }
    root["lm_head"] = {"kernel": state_dict.pop("output.weight").T}
    return root


def remap_phi3_state_dict(state_dict, head_dim=None):
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    n_layers = max([int(k.split('.')[1]) for k in state_dict.keys() if k.startswith("layers.")]) + 1
    hidden_size = state_dict['embed_tokens.weight'].shape[1]
    if head_dim is None:
        head_dim = 128
    n_heads = hidden_size  // head_dim
    n_kv_heads = (state_dict['layers.0.self_attn.qkv_proj.weight'].shape[0] // head_dim - n_heads) // 2

    root = {}
    root["wte"] = {"embedding": state_dict.pop("embed_tokens.weight")}

    # TransformerBlock
    for d in range(n_layers):
        block_d = {}
        block_d["ln_1"] = {
            "scale": state_dict.pop(f"layers.{d}.input_layernorm.weight"),
        }
        w_qkv = state_dict[f"layers.{d}.self_attn.qkv_proj.weight"].T
        w_qkv = w_qkv.reshape(hidden_size, n_heads + 2 * n_kv_heads, head_dim)
        block_d["attn"] = {
            "query": {
                "kernel": w_qkv[:, :n_heads]
            },
            "key": {
                "kernel": w_qkv[:, n_heads:(n_heads + n_kv_heads)]
            },
            "value": {
                "kernel": w_qkv[:, (n_heads + n_kv_heads):]
            },
            "out": {
                "kernel": state_dict.pop(
                    f"layers.{d}.self_attn.o_proj.weight").T.reshape(n_heads, head_dim, hidden_size),
            },
        }
        block_d["ln_2"] = {
            "scale": state_dict.pop(f"layers.{d}.post_attention_layernorm.weight"),
        }

        w_gate_up = state_dict[f"layers.{d}.mlp.gate_up_proj.weight"].T
        c = w_gate_up.shape[1] // 2
        block_d["mlp"] = {
            "gate": {"kernel": w_gate_up[:, :c]},
            "up": {"kernel": w_gate_up[:, c:]},
            "down": {"kernel": state_dict.pop(f"layers.{d}.mlp.down_proj.weight").T},
        }
        root[f"h_{d}"] = block_d

    root["ln_f"] = {
        "scale": state_dict.pop("norm.weight"),
    }
    root["lm_head"] = {"kernel": state_dict.pop("lm_head.weight").T}
    return root
