import functools
from typing import Any, Callable

import jax.numpy as jnp

from flax import struct
from flax import linen as nn

from haxllm.model.modules import SelfAttention, MlpBlock
from haxllm.model.utils import load_config as _load_config


config_hub = {
    "gpt2": dict(
        hidden_size=768,
        n_heads=12,
        n_layers=12,
    ),
    "gpt2-medium": dict(
        hidden_size=1024,
        n_heads=16,
        n_layers=24,
    ),
    "gpt2-large": dict(
        hidden_size=1280,
        n_heads=20,
        n_layers=36,
    ),
    "gpt2-xl": dict(
        hidden_size=1600,
        n_heads=25,
        n_layers=48,
    ),
}

def load_config(name, **kwargs):
    if name in config_hub:
        config = config_hub[name]
    else:
        raise ValueError(f"Unknown llama model {name}")
    return _load_config(TransformerConfig, config, **kwargs)



@struct.dataclass
class TransformerConfig:
    vocab_size: int = 50257
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    hidden_size: int = 768
    n_heads: int = 12
    n_layers: int = 12
    layer_norm_epsilon: float = 1e-5
    n_positions: int = 1024
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    pad_token_id: int = 50256
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    remat: bool = False
    scan: bool = False
    remat_scan: bool = False

    def scan_layers(self):
        return self.n_layers if self.scan else 0

    def remat_scan_lengths(self):
        return (self.n_layers, 1) if self.remat_scan else None

    def scan_lengths(self):
        if self.remat_scan:
            return self.remat_scan_lengths()
        elif self.scan:
            return self.scan_layers()
        else:
            return None


class TransformerBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool
    scan: bool = False

    @nn.compact
    def __call__(self, x):
        inputs, attn_mask = x
        config = self.config

        casual_mask = nn.make_causal_mask(attn_mask, dtype=attn_mask.dtype)
        attn_mask_ = attn_mask[:, None, None, :]
        attn_mask_ = nn.combine_masks(casual_mask, attn_mask_)

        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_1')(inputs)
        x = SelfAttention(
            num_heads=config.n_heads,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            broadcast_dropout=False,
            dropout_rate=config.attn_pdrop,
            deterministic=self.deterministic,
            name='attn')(x, attn_mask_)
        x = nn.Dropout(rate=config.resid_pdrop)(x, deterministic=self.deterministic)
        x = x + inputs

        y = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_2')(x)
        y = MlpBlock(
            activation='gelu_new',
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name='mlp')(y)
        y = nn.Dropout(rate=config.resid_pdrop)(y, deterministic=self.deterministic)
        y = x + y
        
        if self.scan:
            return (y, attn_mask), None
        else:    
            return y, attn_mask
    

class TransformerModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train):
        config = self.config
        remat = config.remat or config.remat_scan

        position_ids = jnp.arange(0, inputs.shape[-1], dtype=jnp.int32)[None]

        embed_layer = nn.remat(nn.Embed) if remat else nn.Embed
        embed_layer = functools.partial(
            embed_layer, dtype=config.dtype, param_dtype=config.param_dtype)
        inputs_embeds = embed_layer(
            num_embeddings=config.vocab_size, features=config.hidden_size, name='wte')(inputs)
        position_embeds = embed_layer(
            num_embeddings=config.n_positions, features=config.hidden_size, name='wpe')(position_ids)

        x = inputs_embeds + position_embeds

        dropout_layer = nn.remat(nn.Dropout, static_argnums=(2,)) if remat else nn.Dropout
        x = dropout_layer(rate=config.embd_pdrop)(x, not train)

        if config.remat_scan:
            remat_scan_lengths = config.remat_scan_lengths()
            TransformerBlockStack = nn.remat_scan(TransformerBlock, lengths=remat_scan_lengths)
            x = TransformerBlockStack(config, deterministic=not train, name='hs')((x, attn_mask))[0]
        else:
            block_fn = TransformerBlock
            if config.remat:
                block_fn = nn.remat(TransformerBlock, prevent_cse=not config.scan)
            if config.scan:
                TransformerBlockStack = nn.scan(block_fn, length=config.scan_layers(), variable_axes={True: 0}, split_rngs={True: True})
                x = TransformerBlockStack(config, deterministic=not train, scan=True, name='hs')((x, attn_mask))[0][0]
            else:
                for d in range(config.n_layers):
                    x = block_fn(config, deterministic=not train, name=f'h_{d}')((x, attn_mask))[0]
        
        norm_layer = nn.remat(nn.LayerNorm) if remat else nn.LayerNorm
        x = norm_layer(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_f')(x)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train=False):
        config = self.config
        x = TransformerModel(config=config, name='transformer')(inputs=inputs, attn_mask=attn_mask, train=train)

        batch_size = inputs.shape[0]
        seq_len = (jnp.not_equal(inputs, config.pad_token_id).sum(-1) - 1)
        x = x[jnp.arange(batch_size), seq_len]

        x = nn.Dense(
            config.num_labels,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name='score')(x)
        return x


def remap_state_dict(state_dict, config: TransformerConfig):
    # Embedding
    root = {}
    root['wte'] = {'embedding': state_dict.pop('wte.weight')}
    root['wpe'] = {'embedding': state_dict.pop('wpe.weight')}
    hidden_size = config.hidden_size
    n_heads = config.n_heads

    # TransformerBlock
    for d in range(config.n_layers):
        block_d = {}
        block_d['ln_1'] = {'scale': state_dict.pop(f'h.{d}.ln_1.weight'), 'bias': state_dict.pop(f'h.{d}.ln_1.bias')}
        c_attn_weight = state_dict[f'h.{d}.attn.c_attn.weight']
        c_attn_bias = state_dict[f'h.{d}.attn.c_attn.bias']
        block_d['attn'] = {
            'query': {
                'kernel': c_attn_weight[:, 0:hidden_size].reshape(hidden_size, n_heads, -1),
                'bias': c_attn_bias[0:hidden_size].reshape(n_heads, -1),
            },
            'key': {
                'kernel': c_attn_weight[:, hidden_size:hidden_size*2].reshape(hidden_size, n_heads, -1),
                'bias': c_attn_bias[hidden_size:hidden_size*2].reshape(n_heads, -1),
            },
            'value': {
                'kernel': c_attn_weight[:, hidden_size*2:hidden_size*3].reshape(hidden_size, n_heads, -1),
                'bias': c_attn_bias[hidden_size*2:hidden_size*3].reshape(n_heads, -1),
            },
            'out': {
                'kernel': state_dict.pop(f'h.{d}.attn.c_proj.weight').reshape(n_heads, -1, hidden_size),
                'bias': state_dict.pop(f'h.{d}.attn.c_proj.bias'),
            },
        }
        block_d['ln_2'] = {'scale': state_dict.pop(f'h.{d}.ln_2.weight'), 'bias': state_dict.pop(f'h.{d}.ln_2.bias')}
        block_d['mlp'] = {
            'fc_1': {
                'kernel': state_dict.pop(f'h.{d}.mlp.c_fc.weight'),
                'bias': state_dict.pop(f'h.{d}.mlp.c_fc.bias'),
            },
            'fc_2': {
                'kernel': state_dict.pop(f'h.{d}.mlp.c_proj.weight'),
                'bias': state_dict.pop(f'h.{d}.mlp.c_proj.bias'),
            },
        }
        root[f'h_{d}'] = block_d

    # Final LayerNorm
    root['ln_f'] = {'scale': state_dict.pop('ln_f.weight'), 'bias': state_dict.pop('ln_f.bias')}
    return root