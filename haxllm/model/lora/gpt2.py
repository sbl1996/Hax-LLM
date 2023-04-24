import math
from typing import Any, Callable, Optional, Tuple, Sequence

import jax.numpy as jnp
import flax.linen as nn
from flax import struct

from haxllm.model.lora.modules import SelfAttention 


def convert_config(config, **kwargs):
    d = {}
    for k in TransformerConfig.__annotations__.keys():
        if hasattr(config, k):
            v = getattr(config, k)
            if v is not None:
                d[k] = v
    for k, v in kwargs.items():
        d[k] = v
    return TransformerConfig(**d)


@struct.dataclass
class TransformerConfig:
    vocab_size: int = 50257
    num_labels: int = 2
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    layer_norm_epsilon: float = 1e-5
    n_positions: int = 1024
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    pad_token_id: int = 50256
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    is_casual: bool = True
    remat: bool = False
    scan_layers: Optional[int] = None
    remat_scan_lengths: Optional[Tuple[int, int]] = None
    attn_lora_r: Optional[Sequence[int]] = (0, 0, 0, 0)
    lora_alpha: int = 1


class MlpBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, deterministic=True):
        config = self.config
        n_inner = config.n_embd * 4

        actual_out_dim = inputs.shape[-1]
        x = nn.Dense(
            n_inner,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name="fc_1")(inputs)
        x = nn.gelu(x)
        x = nn.Dense(
            actual_out_dim,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name="fc_2")(x)
        x = nn.Dropout(rate=config.resid_pdrop)(x, deterministic=deterministic)
        return x


class TransformerBlock(nn.Module):
    config: TransformerConfig
    deterministic: bool
    scan: bool = False

    @nn.compact
    def __call__(self, x):
        inputs, attn_mask = x
        config = self.config

        assert inputs.ndim == 3
        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_1')(inputs)
        x = SelfAttention(
            num_heads=config.n_head,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            qkv_features=config.n_embd,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=True,
            broadcast_dropout=False,
            dropout_rate=config.attn_pdrop,
            deterministic=self.deterministic,
            attn_lora_r=config.attn_lora_r,
            lora_alpha=config.lora_alpha,
            name='attn')(x, attn_mask)
        x = x + inputs

        y = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_2')(x)
        y = MlpBlock(config=config, name='mlp')(y, deterministic=self.deterministic)
        if self.scan:
            return (x + y, attn_mask), None
        else:    
            return x + y, attn_mask


class TransformerModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train):
        config = self.config

        position_ids = jnp.arange(0, inputs.shape[-1], dtype=jnp.int32)[None]

        inputs_embeds = nn.Embed(
            num_embeddings=config.vocab_size, features=config.n_embd, dtype=config.dtype, name='wte')(inputs)
        position_embeds = nn.Embed(
            num_embeddings=config.n_positions, features=config.n_embd, dtype=config.dtype, name='wpe')(position_ids)

        x = inputs_embeds + position_embeds

        x = nn.Dropout(rate=config.embd_pdrop)(x, deterministic=not train)

        if attn_mask is not None:
            if config.is_casual:
                casual_mask = nn.make_causal_mask(attn_mask, dtype=attn_mask.dtype)
                attn_mask = attn_mask[:, None, None, :]
                attn_mask = nn.combine_masks(casual_mask, attn_mask)
            else:
                attn_mask = attn_mask[:, None, None, :]

        if config.remat_scan_lengths is not None:
            remat_scan_layers = math.prod(config.remat_scan_lengths)
            d = config.n_layer - remat_scan_layers
            if d < 0:
                raise ValueError(f"remat_scan_lengths={config.remat_scan_lengths} is too large for n_layer={config.n_layer}")
            for i in range(d):
                x = TransformerBlock(config, deterministic=not train, name=f'h_{i}')((x, attn_mask))[0]
            TransformerBlockStack = nn.remat_scan(TransformerBlock, lengths=config.remat_scan_lengths)
            x = TransformerBlockStack(config, deterministic=not train, name='hs')((x, attn_mask))[0]
        else:
            block_fn = TransformerBlock
            if config.remat:
                block_fn = nn.remat(block_fn)
            scan_layers = config.scan_layers
            d = config.n_layer - scan_layers
            if d < 0:
                raise ValueError(f"scan_layers={config.scan_layers} is too large for n_layer={config.n_layer}")
            for i in range(d):
                x = block_fn(config, deterministic=not train, name=f'h_{i}')((x, attn_mask))[0]
            
            TransformerBlockStack = nn.scan(block_fn, length=scan_layers, variable_axes={True: 0}, split_rngs={True: True})
            x = TransformerBlockStack(config, deterministic=not train, scan=True, name='hs')((x, attn_mask))[0][0]
        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_f')(x)
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
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name='score')(x)
        return x


def remap_state_dict(state_dict, config: TransformerConfig):
    # Embedding
    root = {}
    root['wte'] = {'embedding': state_dict.pop('wte.weight')}
    root['wpe'] = {'embedding': state_dict.pop('wpe.weight')}
    hidden_size = config.n_embd
    n_heads = config.n_head

    # TransformerBlock
    for d in range(config.n_layer):
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