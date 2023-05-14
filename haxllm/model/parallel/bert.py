import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from flax import struct
from flax import linen as nn


def convert_config(config, **kwargs):
    d = dict(
        vocab_size=config.vocab_size,
        type_vocab_size=config.type_vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        n_heads=config.num_attention_heads,
        n_layers=config.num_hidden_layers,
        n_positions=config.max_position_embeddings,
        embd_pdrop=config.hidden_dropout_prob,
        attn_pdrop=config.attention_probs_dropout_prob,
        resid_pdrop=config.hidden_dropout_prob,
        pad_token_id=config.pad_token_id,
    )
    for k, v in kwargs.items():
        d[k] = v
    return TransformerConfig(**d)


@struct.dataclass
class TransformerConfig:
    vocab_size: int = 30522
    type_vocab_size: int = 2
    num_labels: int = 2
    dtype: Any = jnp.float32
    hidden_size: int = 768
    intermediate_size: int = 3072
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
    is_casual: bool = False
    remat: bool = False
    remat_scan_lengths: Optional[Tuple[int, int]] = None


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        intermediate_size = config.intermediate_size
        self.sow
        actual_out_dim = inputs.shape[-1]
        x = nn.Dense(
            intermediate_size,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name="fc_1")(inputs)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(
            actual_out_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name="fc_2")(x)
        return x



class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, attn_mask, deterministic):
        config = self.config

        assert inputs.ndim == 3
        x = nn.SelfAttention(
            num_heads=config.num_attention_heads,
            dtype=config.dtype,
            qkv_features=config.hidden_size,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=True,
            broadcast_dropout=False,
            dropout_rate=config.attn_pdrop,
            deterministic=deterministic,
            name='attn')(inputs, attn_mask)
        x = nn.Dropout(rate=config.resid_pdrop)(x, deterministic=deterministic)
        x = x + inputs
        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_1')(x)

        y = MlpBlock(config=config, name='mlp')(x)
        y = nn.Dropout(rate=config.resid_pdrop)(y, deterministic=deterministic)
        y = x + y
        y = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_2')(y)
        return y


class TransformerBlock2(nn.Module):
    config: TransformerConfig
    deterministic: bool

    @nn.compact
    def __call__(self, x):
        inputs, attn_mask = x
        config = self.config

        assert inputs.ndim == 3
        x = nn.SelfAttention(
            num_heads=config.num_attention_heads,
            dtype=config.dtype,
            qkv_features=config.hidden_size,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=True,
            broadcast_dropout=False,
            dropout_rate=config.attn_pdrop,
            deterministic=self.deterministic,
            name='attn')(inputs, attn_mask)
        x = nn.Dropout(rate=config.resid_pdrop)(x, deterministic=self.deterministic)
        x = x + inputs
        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_1')(x)

        y = MlpBlock(config=config, name='mlp')(x)
        y = nn.Dropout(rate=config.resid_pdrop)(y, deterministic=self.deterministic)
        y = x + y
        y = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln_2')(y)
        return y, attn_mask
    

class TransformerModel(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train):
        config = self.config

        position_ids = jnp.arange(0, inputs.shape[-1], dtype=jnp.int32)[None]
        type_token_ids = jnp.zeros_like(position_ids, dtype=jnp.int32)

        inputs_embeds = nn.Embed(
            num_embeddings=config.vocab_size, features=config.hidden_size, dtype=config.dtype, name='word_embeddings')(inputs)
        position_embeds = nn.Embed(
            num_embeddings=config.n_positions, features=config.hidden_size, dtype=config.dtype, name='position_embeddings')(position_ids)
        token_type_embeds = nn.Embed(
            num_embeddings=config.type_vocab_size, features=config.hidden_size, dtype=config.dtype, name='token_type_embeddings')(type_token_ids)
            
        x = inputs_embeds + position_embeds + token_type_embeds
        x = nn.LayerNorm(epsilon=config.layer_norm_epsilon, dtype=config.dtype, name='ln')(x)
        x = nn.Dropout(rate=config.embd_pdrop)(x, deterministic=not train)

        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None, :]

        if config.remat_scan_lengths is not None:
            remat_scan_layers = math.prod(config.remat_scan_lengths)
            d = config.n_layers - remat_scan_layers
            if d < 0:
                raise ValueError(f"remat_scan_lengths={config.remat_scan_lengths} is too large for n_layers={config.n_layers}")
            for i in range(d):
                x = TransformerBlock(config, name=f'h_{i}')(x, attn_mask, not train)
            TransformerBlockStack = nn.remat_scan(TransformerBlock2, lengths=config.remat_scan_lengths)
            x = TransformerBlockStack(config, deterministic=not train, name='hs')((x, attn_mask))[0]
        else:
            if config.remat and train:
                CheckpointTransformerBlock = nn.remat(TransformerBlock, static_argnums=(3,))
                block_fn = CheckpointTransformerBlock
            else:
                block_fn = TransformerBlock
            for i in range(config.n_layers):
                x = block_fn(config, name=f'h_{i}')(x, attn_mask, not train)
        return x


class TransformerSequenceClassifier(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, *, inputs, attn_mask, train):
        config = self.config
        x = TransformerModel(config=config, name='transformer')(inputs=inputs, attn_mask=attn_mask, train=train)
        x = x[:, 0]
        x = nn.Dense(
            config.num_labels,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            name='score')(x)
        return x


def remap_state_dict(state_dict, config: TransformerConfig):
    # Embedding
    state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items()}
    root = {}
    root['word_embeddings'] = {'embedding': state_dict.pop('embeddings.word_embeddings.weight')}
    root['position_embeddings'] = {'embedding': state_dict.pop('embeddings.position_embeddings.weight')}
    root['token_type_embeddings'] = {'embedding': state_dict.pop('embeddings.token_type_embeddings.weight')}
    root['ln'] = {'scale': state_dict.pop('embeddings.LayerNorm.gamma'), 'bias': state_dict.pop('embeddings.LayerNorm.beta')}
    hidden_size = config.hidden_size
    n_heads = config.num_attention_heads

    # TransformerBlock
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

    return root