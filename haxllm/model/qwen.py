from typing import Callable, Any

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.modules import RMSNorm, make_block_stack
from haxllm.model.parallel import GLUMlpBlock, DenseGeneral, SelfAttention, Embed
from haxllm.model.utils import load_config as _load_config
from haxllm.model.mixin import RematScanConfigMixin


config_hub = {
    "qwen-7b": dict(
        hidden_size=4096,
        intermediate_size=11008,
        n_heads=32,
        n_layers=32,
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
    rms_norm_eps: float = 1e-5
    n_positions: int = 2048
    pad_token_id: int = 151850  # last id of unpadded vocab, unpossible to be used (<extra_199>)
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    decode: bool = False
    memory_efficient_attention: bool = False
    shard: bool = False
    shard_cache: bool = False


class TransformerBlock(nn.Module):
    config: TransformerConfig
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        if config.memory_efficient_attention or config.decode:
            mask = None
        else:
            mask = nn.make_causal_mask(inputs[..., 0], dtype=jnp.bool_)

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
            memory_efficient=config.memory_efficient_attention,
            memory_efficient_mask_mode='causal',
            rope=True,
            query_shard_axes=("X", "Y", None),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            shard_cache=config.shard_cache,
            name="attn")(x, mask)

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
        if self.scan:
            return x + y, None
        else:
            return x + y


class TransformerModel(nn.Module):
    config: TransformerConfig
    block_cls: Callable = TransformerBlock

    @nn.compact
    def __call__(self, inputs, train):
        config = self.config
        remat = config.remat or config.remat_scan

        embed_layer = Embed if remat else Embed
        x = embed_layer(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            shard_axes={"embedding": (None, "Y")},
            shard=config.shard,
            name="wte")(inputs)

        x = make_block_stack(
            self.block_cls, config.n_layers, config)(x, train)

        norm_layer = RMSNorm if remat else RMSNorm
        x = norm_layer(epsilon=config.rms_norm_eps,
                       dtype=config.dtype, name="ln_f")(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(
            config=config, name="transformer")(inputs, train)

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
