from typing import Callable, Any, Optional

import enum

import jax.numpy as jnp

import flax.linen as nn
from flax import struct

from haxllm.model.quantize import QConfig
from haxllm.gconfig import get_remat_policy
from haxllm.model.modules import RMSNorm
from haxllm.model.parallel import GLUMlpBlock, remat, Embed, SelfAttention
from haxllm.model.mixin import RematScanConfigMixin, RoPEScalingConfigMixin
from haxllm.chat.setting import register_chat_setting
from haxllm.model.llama import remap_llama_state_dict


class QueryPreAttentionNormalization(enum.Enum):
  """Initialization strategy."""

  # Whether to scale the query by 1/sqrt(head_dim)
  BY_ONE_OVER_SQRT_HEAD_DIM = enum.auto()

  # Whether to scale the query by `embed_dim // num_heads`
  BY_EMBED_DIM_DIV_NUM_HEADS = enum.auto()

  # Whether to scale the query by `1/sqrt(embed_dim // num_heads)`
  BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS = enum.auto()


def Gemma2Config(**kwargs):
    base = dict(
        n_positions=8192,
    )
    return {**base, **kwargs}


config_hub = {
    "gemma2-2b": Gemma2Config(
        hidden_size=2304,
        intermediate_size=9216,
        n_heads=8,
        n_kv_heads=4,
        head_dim=256,
        n_layers=26,
    ),
    "gemma2-9b": Gemma2Config(
        hidden_size=3584,
        intermediate_size=14336,
        n_heads=16,
        n_kv_heads=8,
        head_dim=256,
        n_layers=42,
    ),
    "gemma2-27b": Gemma2Config(
        hidden_size=4608,
        intermediate_size=36864,
        n_heads=32,
        n_kv_heads=16,
        head_dim=128,
        n_layers=46,
        query_pre_attn_norm=QueryPreAttentionNormalization.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS,
    ),
}


@struct.dataclass
class TransformerConfig(RematScanConfigMixin, RoPEScalingConfigMixin):
    vocab_size: int = 256000
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 2304
    intermediate_size: int = 9216
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 256
    n_layers: int = 26
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    n_positions: int = 8192
    final_logit_softcap: float = 30.0
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True
    query_pre_attn_norm: QueryPreAttentionNormalization = (
        QueryPreAttentionNormalization.BY_ONE_OVER_SQRT_HEAD_DIM
    )
    attn_logits_soft_cap: float = 50.0
    sliding_window_size: int = 4096
    pad_token_id: int = 3
    bos_token_id: int = 2
    eos_token_id: int = 1
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    padding_left: bool = False
    decode: bool = False
    shard: bool = False
    shard_cache: bool = False
    qconfig: Optional[QConfig] = None

    def query_pre_attn_scalar(self) -> float:
        """Returns the scalar to multiply the query by before attention."""
        match self.query_pre_attn_norm:
            case QueryPreAttentionNormalization.BY_EMBED_DIM_DIV_NUM_HEADS:
                return self.hidden_size // self.n_heads
            case QueryPreAttentionNormalization.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS:  # pylint: disable=line-too-long
                return (self.hidden_size // self.n_heads)**-0.5
            case QueryPreAttentionNormalization.BY_ONE_OVER_SQRT_HEAD_DIM | _:
                return self.head_dim**-0.5


class TransformerBlock(nn.Module):
    config: TransformerConfig
    is_sliding: bool = False
    scan: bool = False

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        inputs, padding_mask = inputs

        x = RMSNorm(epsilon=config.rms_norm_eps, offset=1.0,
                    dtype=config.dtype, name="ln_1")(inputs)
        x = SelfAttention(
            head_dim=config.head_dim,
            num_heads=config.n_heads,
            num_kv_heads=config.n_kv_heads,
            sliding_window_size=config.sliding_window_size if self.is_sliding else None,
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
            scale=config.query_pre_attn_scalar(),
            attn_logits_soft_cap=config.attn_logits_soft_cap,
            query_shard_axes=("X", "Y", None),
            kv_shard_axes=("X", "Y", None),
            kv_cache_shard_axes=(None, "X", "Y", None),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            shard_cache=config.shard_cache,
            qconfig=config.qconfig,
            name="attn")(x, padding_mask=padding_mask)
        if config.use_post_attn_norm:
            x = RMSNorm(epsilon=config.rms_norm_eps, offset=1.0,
                        dtype=config.dtype, name="ln_1p")(x)

        x = x + inputs

        y = RMSNorm(epsilon=config.rms_norm_eps, offset=1.0,
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
            activation="gelu",
            name="mlp")(y)
        if config.use_post_ffw_norm:
            y = RMSNorm(epsilon=config.rms_norm_eps, offset=1.0,
                        dtype=config.dtype, name="ln_2p")(y)

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
        is_remat = config.remat or config.remat_scan
        x, padding_mask = inputs

        block_fn = self.block_cls
        if is_remat and train:
            remat_policy = get_remat_policy()
            block_fn = remat(block_fn, policy=remat_policy)
        for i in range(config.n_layers):
            x = block_fn(
                config=config, is_sliding=i % 2 == 0, name=f"h_{i}")((x, padding_mask))[0]

        x = RMSNorm(epsilon=config.rms_norm_eps, offset=1.0,
                       dtype=config.dtype, name="ln_f")(x)
        return x


class TransformerLMHeadModel(nn.Module):
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
        x *= jnp.sqrt(config.hidden_size).astype(x.dtype)

        padding_mask = None
        if config.decode and config.padding_left and input_ids.shape[1] > 1:
            padding_mask = jnp.equal(input_ids, config.pad_token_id)

        x = TransformerModel(
            config=config, name="transformer")(inputs=(x, padding_mask), train=train)

        x = embed_layer.attend(x)
        if config.final_logit_softcap is not None:
            x /= config.final_logit_softcap
            x = jnp.tanh(x) * config.final_logit_softcap
        return x


remap_gemma_state_dict = remap_llama_state_dict


@register_chat_setting()
class Gemma2ChatSetting:
    name = "gemma2"
    system = ""
    roles = ("user", "model")
    stop_token_ids = (1, 107)

    def get_prompt(self, messages):
        if messages[0][0] == "system":
            raise ValueError("System role not supported")            
        # ret = "<bos>"
        ret = ""
        for i, (role, message) in enumerate(messages):
            if (role == self.roles[0]) != (i % 2 == 0):
                raise ValueError("Conversation roles must alternate user/assistant/user/assistant/...")
            ret += f"<start_of_turn>{role}\n"
            if message:
                ret += f"{message.strip()}<end_of_turn>\n"
            else:
                assert i == len(messages) - 1 and role == self.roles[1]
        return ret