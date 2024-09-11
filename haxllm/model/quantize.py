from functools import partial
import enum
from dataclasses import dataclass
from typing import Sequence, Optional

import json

import numpy as np
import jax
import jax.numpy as jnp


class QuantSource(enum.Enum):
    half = 0
    autogptq_q8 = 1
    autoawq_q4 = 2
    autogptq_q4 = 3


class QuantMethod(enum.Enum):
    rtn_q8_0 = 0
    native_q4 = 1


dtype_to_bits = {
    jnp.float32: 32,
    jnp.float16: 16,
    jnp.bfloat16: 16,
    jnp.int8: 8,
    jnp.int32: 32,
    jnp.uint4: 4,
}

@dataclass
class QConfig:
    source: QuantSource = QuantSource.half
    method: QuantMethod = QuantMethod.rtn_q8_0
    group_size: int = 128
    sym: bool = False
    q_layers: Sequence[str] = ("attn.out", "mlp.gate", "mlp.up", "mlp.down")
    use_g_idx: bool = False
    p_dtype: Optional[jnp.dtype] = None

    def __post_init__(self):
        for l in self.q_layers:
            assert l in ("attn.query", "attn.key", "attn.value", "attn.out", "mlp.gate", "mlp.up", "mlp.down"), f"Invalid layer: {l}"        

        if self.source in [QuantSource.autoawq_q4, QuantSource.autogptq_q4]:
            assert self.method in [QuantMethod.native_q4]
        if self.source in [QuantSource.autogptq_q8]:
            assert self.method == QuantMethod.rtn_q8_0

        if self.source in [QuantSource.autogptq_q4] or self.method in [QuantMethod.rtn_q8_0]:
            assert self.sym
        elif self.source in [QuantSource.autoawq_q4]:
            assert not self.sym

        if self.use_g_idx:
            assert self.source == QuantSource.autogptq_q4

    @property
    def f_bits(self):
        return dtype_to_bits[self.f_dtype]

    @property
    def f_dtype(self):
        # dtype in the dumped safetensors file
        if self.method == QuantMethod.rtn_q8_0:
            return jnp.int8
        elif self.method == QuantMethod.native_q4:
            return jnp.int32
        else:
            raise NotImplementedError
    
    @property
    def w_bits(self):
        return dtype_to_bits[self.w_dtype]

    @property
    def w_dtype(self):
        # dtype of quantized model weight
        if self.p_dtype is not None:
            return self.p_dtype
        if self.method == QuantMethod.rtn_q8_0:
            return jnp.int8
        elif self.method == QuantMethod.native_q4:
            return jnp.uint4
        else:
            raise NotImplementedError
    
    def quantize(self, *args):
        if self.method == QuantMethod.rtn_q8_0:
            assert self.source == QuantSource.half
            w, = args
            w, scale = group_abs_max_int8_quantize(w, group_size=self.group_size)
            return w, scale
        else:
            raise NotImplementedError

    def dequantize(self, q):
        if self.method == QuantMethod.rtn_q8_0:
            return group_dequantize(q["qweight"], None, q["scales"])
        elif self.method == QuantMethod.native_q4:
            scales = q["scales"]
            qweight = q["qweight"].astype(scales.dtype)
            if self.source == QuantSource.autoawq_q4:
                qzeros = q["zeros"][:, None]
                g_idx = None
            elif self.source == QuantSource.autogptq_q4:
                qzeros = 8
                g_idx = q.get("g_idx", None)
            return group_dequantize(qweight, qzeros, scales, g_idx)
        else:
            raise NotImplementedError

    def _pack(self, w):
        return pack_q4(w)
    
    def _unpack(self, w):
        return unpack_q4(w)

    def requantize(self, *args):
        if self.source == QuantSource.autoawq_q4:
            qweight, qzeros = args
            weight, zeros = jax.tree.map(
                lambda x: int_unpack(jnp.array(x)), (qweight, qzeros))
            weight = np.array(self._pack(weight))
            zeros = np.array(zeros)
            return weight, zeros
        elif self.source == QuantSource.autogptq_q4:
            qweight, qzeros = args
            weight = int_unpack(jnp.array(qweight), 4, 'gptq')
            weight = np.array(self._pack(weight))
            return weight, qzeros
        elif self.source == QuantSource.autogptq_q8:
            qweight, qzeros = args
            weight = int_unpack(jnp.array(qweight), 8, 'gptq') - 128
            weight = np.array(weight)
            return weight, qzeros
        else:
            raise NotImplementedError

    def __str__(self):
        return f"QConfig(" \
               f"method={self.method}, " \
               f"source={self.source}, " \
               f"group_size={self.group_size}, " \
               f"sym={self.sym}, "\
               f"p_dtype={self.p_dtype}, " \
               f"use_g_idx={self.use_g_idx}, " \
               f"q_layers={self.q_layers})"

    def to_json(self):
        return json.dumps(
            {
                "source": self.source.name,
                "method": self.method.name,
                "sym": self.sym,
                "group_size": self.group_size,
                "p_dtype": self.p_dtype.dtype.name if self.p_dtype is not None else None,
                "q_layers": self.q_layers,
                "use_g_idx": self.use_g_idx,
            }, indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str):
        data = json.loads(json_str)
        p_dtype = data.get("p_dtype", None)
        kwargs = dict(
            source=QuantSource[data["source"]],
            method=QuantMethod[data["method"]],
            sym=data["sym"],
            p_dtype=jnp.dtype(p_dtype) if p_dtype is not None else None,
            q_layers=tuple(data["q_layers"]),
        )
        if "group_size" in data:
            kwargs["group_size"] = data["group_size"]
        if "use_g_idx" in data:
            kwargs["use_g_idx"] = data["use_g_idx"]
        return cls(**kwargs)


def np_roundf(n):
    a = jnp.abs(n)
    floored = jnp.floor(a)
    b = floored + jnp.floor(2 * (a - floored))
    return jnp.sign(n) * b


@partial(jax.jit, static_argnums=(1,))
def abs_max_int8_quantize(x, axis=-1):
    """
    Perform abs max int8 quantization on the input array.
    
    Args:
    x (np.ndarray): Input array to be quantized.
    
    Returns:
    tuple: Quantized array (int8), scale factor (float)
    """
    x = x.astype(jnp.float32, copy=False)
    d = jnp.abs(x).max(axis=axis, keepdims=True) / 127
    id = jnp.where(d == 0, 0, 1 / d)
    qs = np_roundf(x * id)
    qs = qs.astype(jnp.int8)
    return qs, d.squeeze(axis)


def group_abs_max_int8_quantize(x, group_size, q_dtype=None):
    if q_dtype is None:
        q_dtype = x.dtype
    x = x.reshape(-1, group_size, *x.shape[1:])
    x = jnp.array(x)
    x, scale = abs_max_int8_quantize(x, axis=1)
    x = np.array(x)
    scale = np.array(scale, dtype=q_dtype)
    x = x.reshape(-1, *x.shape[2:])
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
def int_unpack(x, bits=4, variant='awq'):
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


@jax.jit
def pack_q4(x):
    n = x.shape[0]
    return (x[:n//2] * 16 + x[n//2:]).view(jnp.int32)


@partial(jax.jit, static_argnums=(1,))
def unpack_q4(x, dtype=jnp.uint4):
    assert x.dtype == jnp.int32
    x = jnp.asarray(x)
    x = x.view(jnp.uint8)
    x1 = x >> 4
    x2 = x & 0xf
    x = jnp.concatenate([x1, x2], axis=0)
    x = x.view(jnp.int8)
    x = x.astype(dtype)
    return x


def group_dequantize(qweight, qzeros, scales, g_idx=None):
    group_size = qweight.shape[0] // scales.shape[0]
    qweight = qweight.reshape(-1, group_size, *qweight.shape[1:])
    if qzeros is not None:
        qweight = qweight - qzeros
    if g_idx is not None:
        scales = scales[g_idx].reshape(-1, group_size, *scales.shape[1:])
    else:
        scales = scales[:, None]
    weight = qweight * scales
    weight = weight.reshape(-1, *weight.shape[2:])
    return weight
