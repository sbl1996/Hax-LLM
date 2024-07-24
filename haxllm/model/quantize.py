from functools import partial
import enum
from dataclasses import dataclass
from typing import Sequence, Optional

import json

import numpy as np
import jax
import jax.numpy as jnp


class QuantMethod(enum.Enum):
    rtn_q8_0 = 0
    awq_q4 = 1
    gptq_q4 = 2


@dataclass
class QConfig:
    method: QuantMethod = QuantMethod.rtn_q8_0
    block_size: int = 32
    group_size: int = 128
    sym: bool = False
    q_dtype: Optional[jnp.dtype] = None
    q_layers: Sequence[str] = ("attn.out", "mlp.gate", "mlp.up", "mlp.down")

    def __post_init__(self):
        for l in self.q_layers:
            assert l in ("attn.query", "attn.key", "attn.value", "attn.out", "mlp.gate", "mlp.up", "mlp.down"), f"Invalid layer: {l}"
        if self.method in [QuantMethod.rtn_q8_0, QuantMethod.gptq_q4]:
            assert self.sym
        elif self.method in [QuantMethod.awq_q4]:
            assert not self.sym

    @property
    def w_bits(self):
        if self.method == QuantMethod.rtn_q8_0:
            return 8
        elif self.method == QuantMethod.awq_q4:
            return 4
        elif self.method == QuantMethod.gptq_q4:
            return 4
        else:
            raise NotImplementedError

    @property
    def w_dtype(self):
        if self.method == QuantMethod.rtn_q8_0:
            return jnp.int8
        elif self.method == QuantMethod.awq_q4:
            return jnp.int32
        elif self.method == QuantMethod.gptq_q4:
            return jnp.int32
        else:
            raise NotImplementedError
    
    def quantize(self, *args):
        if self.method == QuantMethod.rtn_q8_0:
            w, = args
            w, scale = block_abs_max_int8_quantize(w, block_size=self.block_size, q_dtype=self.q_dtype)
            return w, scale
        else:
            raise NotImplementedError

    def dequantize(self, q):
        if self.method == QuantMethod.rtn_q8_0:
            return block_dequantize(q["kernel"], q["scales"])
        elif self.method == QuantMethod.awq_q4:
            return dequantize2(q["qweight"], q["zeros"][:, None], q["scales"])
        elif self.method == QuantMethod.gptq_q4:
            return dequantize2(q["qweight"], 8, q["scales"])
        else:
            raise NotImplementedError

    def requantize(self, *args):
        if self.method == QuantMethod.awq_q4:
            qweight, qzeros = args
            weight, zeros = jax.tree.map(
                lambda x: unpack(jnp.array(x)), (qweight, qzeros))
            weight = np.array(pack2(weight))
            zeros = np.array(zeros)
            return weight, zeros
        elif self.method == QuantMethod.gptq_q4:
            qweight, qzeros = args
            weight = unpack(jnp.array(qweight), 4, 'gptq')
            weight = np.array(pack2(weight))
            return weight, qzeros
        else:
            raise NotImplementedError

    def __str__(self):
        if self.method == QuantMethod.rtn_q8_0:
            return f"QConfig(method={self.method}, block_size={self.block_size}, sym={self.sym}, q_dtype={self.q_dtype}), q_layers={self.q_layers}"
        elif self.method == QuantMethod.awq_q4:
            return f"QConfig(method={self.method}, group_size={self.group_size}, sym={self.sym}, q_dtype={self.q_dtype}, q_layers={self.q_layers})"
        elif self.method == QuantMethod.gptq_q4:
            return f"QConfig(method={self.method}, group_size={self.group_size}, sym={self.sym}, q_dtype={self.q_dtype}, q_layers={self.q_layers})"
        else:
            raise NotImplementedError

    def to_json(self):
        return json.dumps(
            {
                "method": self.method.name,
                "sym": self.sym,
                "block_size": self.block_size,
                "group_size": self.group_size,
                "q_dtype": self.q_dtype.dtype.name if self.q_dtype is not None else None,
                "q_layers": self.q_layers,
            }, indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        kwargs = dict(
            method=QuantMethod[data["method"]],
            sym=data["sym"],
            q_dtype=jnp.dtype(data["q_dtype"]) if data["q_dtype"] is not None else None,
            q_layers=tuple(data["q_layers"]),
        )
        if "block_size" in data:
            kwargs["block_size"] = data["block_size"]
        if "group_size" in data:
            kwargs["group_size"] = data["group_size"]
        return cls(**kwargs)


def block_dequantize(x, scale):
    n_blocks = scale.shape[-1]
    block_size = x.shape[-1] // scale.shape[-1]
    original_shape = x.shape
    x = x.reshape(*x.shape[:-1], n_blocks, block_size)
    scale = jnp.expand_dims(scale, axis=-1)
    x = x.astype(scale.dtype) * scale
    return x.reshape(*original_shape)


def np_roundf(n):
    a = jnp.abs(n)
    floored = jnp.floor(a)
    b = floored + jnp.floor(2 * (a - floored))
    return jnp.sign(n) * b


@jax.jit
def jit_abs_max_int8_quantize(x):
    """
    Perform abs max int8 quantization on the input array.
    
    Args:
    x (np.ndarray): Input array to be quantized.
    
    Returns:
    tuple: Quantized array (int8), scale factor (float)
    """
    x = x.astype(jnp.float32, copy=False)
    d = jnp.abs(x).max(axis=-1, keepdims=True) / 127
    id = jnp.where(d == 0, 0, 1 / d)
    qs = np_roundf(x * id)
    qs = qs.astype(jnp.int8)
    return qs, d.squeeze(-1)


def block_abs_max_int8_quantize(x, block_size, q_dtype):
    if q_dtype is None:
        q_dtype = x.dtype
    x = x.reshape(*x.shape[:-1], -1, block_size)
    x = jnp.array(x)
    x, scale = jit_abs_max_int8_quantize(x)
    x = np.array(x)
    scale = np.array(scale, dtype=q_dtype)
    x = x.reshape(*x.shape[:-2], -1)
    return x, scale


def reverse_awq_order(iweights, izeros, bits: int):
    reverse_order_tensor = jnp.arange(
        iweights.shape[-1], dtype=jnp.int32)
    reverse_order_tensor = reverse_order_tensor.reshape(-1, 32 // bits)
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.reshape(-1)

    if izeros is not None:
        izeros = izeros[..., reverse_order_tensor]
    iweights = iweights[..., reverse_order_tensor]

    return iweights, izeros


@partial(jax.jit, static_argnums=(1, 2))
def unpack(x, bits=4, variant='awq'):
    shifts = jnp.arange(0, 32, bits, dtype=x.dtype)
    if variant == 'awq':
        for i in range(x.ndim):
            shifts = jnp.expand_dims(shifts, axis=0)
        axis = -1
    elif variant == 'gptq':
        shifts = shifts[None, :]
        for i in range(x.ndim - 1):
            shifts = jnp.expand_dims(shifts, axis=-1)
        axis = 1
    x = jnp.expand_dims(x, axis=axis)
    x = jnp.bitwise_right_shift(x, shifts).astype(jnp.int8)
    if variant == 'awq':
        x = x.reshape(*x.shape[:-2], -1)
        x = reverse_awq_order(x, None, bits)[0]
    else:
        x = x.reshape(-1, *x.shape[2:])
    x = jnp.bitwise_and(x, (2**bits)-1)
    return x


@partial(jax.jit, static_argnums=(1,))
def pack2(x, bits=4):
    assert bits == 4
    return (x[..., ::2] * 16 + x[..., 1::2]).view(jnp.int32)


@partial(jax.jit, static_argnums=(1,))
def unpack2(x, bits=4):
    assert bits == 4
    x = x.view(jnp.uint8)
    x1, x2 = jnp.divmod(x, 16)
    x = jnp.stack([x1, x2], axis=-1)
    x = x.reshape(*x.shape[:-2], -1)
    x = x.view(jnp.int8)
    return x


def group_dequantize(qweight, qzeros, scales):
    group_size = qweight.shape[0] // scales.shape[0]
    qweight = qweight.reshape(-1, group_size, *qweight.shape[1:])
    weight = (qweight - qzeros) * scales[:, None]
    weight = weight.reshape(-1, *weight.shape[2:])
    return weight


def dequantize2(iweight, qzeros, scales):
    bits = scales.shape[-1] // iweight.shape[-1] // 2
    qweight = unpack2(iweight, bits)
    return group_dequantize(qweight, qzeros, scales)


# def unpack_awq(qweight, qzeros, bits: int):
#     shifts = jnp.arange(0, 32, bits, dtype=qweight.dtype)

#     for i in range(qweight.ndim):
#         shifts = jnp.expand_dims(shifts, axis=0)
#     iweights = jnp.bitwise_right_shift(qweight[..., None], shifts).astype(jnp.int8)
#     iweights = iweights.reshape(*iweights.shape[:-2], -1)

#     if qzeros is not None:
#         izeros = jnp.bitwise_right_shift(qzeros[..., None], shifts).astype(jnp.int8)
#         izeros = izeros.reshape(*izeros.shape[:-2], -1)
#     else:
#         izeros = qzeros

#     return iweights, izeros


# def awq_dequantize(qweight, qzeros, scales):
#     bits = scales.shape[-1] // qweight.shape[-1] // 2
#     # qweight int32, qzeros int32
#     # Unpack the qweight and qzeros tensors
#     iweight, izeros = unpack_awq(qweight, qzeros, bits)
#     # Reverse the order of the iweight and izeros tensors
#     iweight, izeros = reverse_awq_order(iweight, izeros, bits)

#     # overflow checks
#     iweight = jnp.bitwise_and(iweight, (2**bits) - 1)
#     izeros = jnp.bitwise_and(izeros, (2**bits) - 1)

#     # fp16 weights
#     group_size = iweight.shape[0] // scales.shape[0]
#     iweight = iweight.reshape(-1, group_size, *iweight.shape[1:])
#     iweight = (iweight - izeros[:, None]) * scales[:, None]
#     iweight = iweight.reshape(-1, *iweight.shape[2:])

#     return iweight
