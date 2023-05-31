from typing import Any, Callable
import functools

import jax.numpy as jnp

from flax import struct
from flax import linen as nn

from haxllm.model.parallel import SelfAttention, MlpBlock, Embed, PrefixEmbed
from haxllm.model.modules import make_block_stack
from haxllm.model.utils import load_config as _load_config, truncated_normal_init
from haxllm.config_utils import RematScanConfigMixin


config_hub = {
    "bert-base-uncased": dict(
        hidden_size=768,
        intermediate_size=3072,
        n_heads=12,
        n_layers=12,
    ),
    "bert-large-uncased": dict(
        hidden_size=1024,
        intermediate_size=4096,
        n_heads=16,
        n_layers=24,
    ),
    "bert-base-cased": dict(
        hidden_size=768,
        intermediate_size=3072,
        n_heads=12,
        n_layers=12,
    ),
    "bert-large-cased": dict(
        hidden_size=1024,
        intermediate_size=4096,
        n_heads=16,
        n_layers=24,
    ),
    "roberta-large": dict(
        hidden_size=1024,
        intermediate_size=4096,
        n_heads=16,
        n_layers=24,
        vocab_size=50265,
    ),
}

def load_config(name, **kwargs):
    if name in config_hub:
        config = config_hub[name]
    else:
        raise ValueError(f"Unknown llama model {name}")
    return _load_config(TransformerConfig, config, **kwargs)



@struct.dataclass
class TransformerConfig(RematScanConfigMixin):
    vocab_size: int = 30522
    type_vocab_size: int = 2
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    pooler: bool = True
    hidden_size: int = 768
    intermediate_size: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    layer_norm_epsilon: float = 1e-12
    n_positions: int = 512
    initializer_range: float = 0.02
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    cls_pdrop: float = 0.1
    pad_token_id: int = 0
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros_init()
    shard: bool = False
    pre_seq_len: int = 0
    prefix_projection: bool = False
    prefix_hidden_size: int = 512


class TransformerBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool
    scan: bool = False

    @nn.compact
    def __call__(self, x):
        inputs, attn_mask = x
        config = self.config

        past_key_value = PrefixEmbed(
            seq_len=config.pre_seq_len,
            projection=config.prefix_projection,
            prefix_features=config.prefix_hidden_size,
            features=(config.n_heads, config.hidden_size // config.n_heads),
            dtype=config.dtype,
            name="prefix"
        )(inputs)

        if attn_mask is not None:
            # (B, H, Q, KV)
            attn_mask_ = jnp.concatenate(
                (jnp.ones((attn_mask.shape[0], config.pre_seq_len), dtype=attn_mask.dtype), attn_mask), axis=-1)
            attn_mask_ = attn_mask_[:, None, None, :]
        else:
            attn_mask_ = None
        
        x = SelfAttention(
            num_heads=config.n_heads,
            max_len=config.n_positions,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            dropout_rate=config.attn_pdrop,
            deterministic=self.deterministic,
            broadcast_dropout=False,
            qkv_shard_axes=("X", "Y", None),
            out_shard_axes=("Y", None, "X"),
            shard=config.shard,
            name='attn')(inputs, attn_mask_, past_key_value)
        x = nn.Dropout(rate=config.resid_pdrop)(x, deterministic=self.deterministic)
        x = x + inputs
        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_1')(x)

        y = MlpBlock(
            intermediate_size=config.intermediate_size,
            activation='gelu',
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            shard_axes1=("X", "Y"),
            shard_axes2=("Y", "X"),
            shard=config.shard,
            name='mlp')(x)
        y = nn.Dropout(rate=config.resid_pdrop)(y, deterministic=self.deterministic)
        y = x + y
        y = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_2')(y)

        if self.scan:
            return (y, attn_mask), None
        else:    
            return y, attn_mask


class TransformerModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, position_ids=None, train=False):
        config = self.config

        if position_ids is None:
            position_ids = jnp.arange(0, inputs.shape[-1], dtype=jnp.int32)[None]
        position_ids = position_ids + config.pre_seq_len
        type_token_ids = jnp.zeros_like(position_ids, dtype=jnp.int32)

        embed_layer = functools.partial(
            nn.Embed, dtype=config.dtype, param_dtype=config.param_dtype)

        inputs_embeds = embed_layer(
            num_embeddings=config.vocab_size, features=config.hidden_size, name='word_embeddings')(inputs)
        position_embeds = embed_layer(
            num_embeddings=config.n_positions, features=config.hidden_size, name='position_embeddings')(position_ids)
        token_type_embeds = embed_layer(
            num_embeddings=config.type_vocab_size, features=config.hidden_size, name='token_type_embeddings')(type_token_ids)
            
        x = inputs_embeds + position_embeds + token_type_embeds

        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln')(x)

        x = nn.Dropout(rate=config.embd_pdrop)(x, not train)

        block_fn = functools.partial(
            TransformerBlock, deterministic=not train)
        x = make_block_stack(block_fn, config.n_layers, config)((x, attn_mask), train)[0]

        if config.pooler:
            x = nn.Dense(
                config.hidden_size,
                dtype=config.dtype,
                param_dtype=config.param_dtype,
                kernel_init=truncated_normal_init(stddev=config.initializer_range),
                bias_init=config.bias_init,
                name="pooler_fc")(x[:, 0])
            x = jnp.tanh(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(config=config, name='transformer')(inputs=inputs, attn_mask=attn_mask, train=train)

        if not config.pooler:
            x = x[:, 0]
        
        x = nn.Dropout(rate=config.cls_pdrop)(x, not train)
        x = nn.Dense(
            config.num_labels,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=nn.initializers.normal(stddev=config.initializer_range),
            bias_init=config.bias_init,
            name='score')(x)
        return x


def remap_state_dict(state_dict, config: TransformerConfig):
    state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items()}
    root = {}
    root['word_embeddings'] = {'embedding': state_dict.pop('embeddings.word_embeddings.weight')}
    root['position_embeddings'] = {'embedding': state_dict.pop('embeddings.position_embeddings.weight')}
    root['token_type_embeddings'] = {'embedding': state_dict.pop('embeddings.token_type_embeddings.weight')}
    root['ln'] = {'scale': state_dict.pop('embeddings.LayerNorm.gamma'), 'bias': state_dict.pop('embeddings.LayerNorm.beta')}
    hidden_size = config.hidden_size
    n_heads = config.n_heads

    for d in range(config.n_layers):
        block_d = {}
        block_d['attn'] = {
            'query': {
                'kernel': state_dict.pop(f'encoder.layer.{d}.attention.self.query.weight').T.reshape(hidden_size, n_heads, -1),
                'bias': state_dict.pop(f'encoder.layer.{d}.attention.self.query.bias').reshape(n_heads, -1),
            },
            'key': {
                'kernel': state_dict.pop(f'encoder.layer.{d}.attention.self.key.weight').T.reshape(hidden_size, n_heads, -1),
                'bias': state_dict.pop(f'encoder.layer.{d}.attention.self.key.bias').reshape(n_heads, -1),
            },
            'value': {
                'kernel': state_dict.pop(f'encoder.layer.{d}.attention.self.value.weight').T.reshape(hidden_size, n_heads, -1),
                'bias': state_dict.pop(f'encoder.layer.{d}.attention.self.value.bias').reshape(n_heads, -1),
            },
            'out': {
                'kernel': state_dict.pop(f'encoder.layer.{d}.attention.output.dense.weight').T.reshape(n_heads, -1, hidden_size),
                'bias': state_dict.pop(f'encoder.layer.{d}.attention.output.dense.bias'),
            },
        }
        block_d['ln_1'] = {'scale': state_dict.pop(f'encoder.layer.{d}.attention.output.LayerNorm.gamma'),
                           'bias': state_dict.pop(f'encoder.layer.{d}.attention.output.LayerNorm.beta')}

        block_d['mlp'] = {
            'fc_1': {
                'kernel': state_dict.pop(f'encoder.layer.{d}.intermediate.dense.weight').T,
                'bias': state_dict.pop(f'encoder.layer.{d}.intermediate.dense.bias'),
            },
            'fc_2': {
                'kernel': state_dict.pop(f'encoder.layer.{d}.output.dense.weight').T,
                'bias': state_dict.pop(f'encoder.layer.{d}.output.dense.bias'),
            },
        }
        block_d['ln_2'] = {'scale': state_dict.pop(f'encoder.layer.{d}.output.LayerNorm.gamma'),
                           'bias': state_dict.pop(f'encoder.layer.{d}.output.LayerNorm.beta')}
        root[f'h_{d}'] = block_d

    root['pooler_fc'] = {'kernel': state_dict.pop('pooler.dense.weight').T, 'bias': state_dict.pop('pooler.dense.bias')}

    return root