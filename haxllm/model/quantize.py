import numpy as np

def abs_max_int8_quantize(x):
    """
    Perform abs max int8 quantization on the input array.
    
    Args:
    x (np.ndarray): Input array to be quantized.
    
    Returns:
    tuple: Quantized array (int8), scale factor (float)
    """
    dtype = x.dtype
    abs_max = np.max(np.abs(x), axis=-1, keepdims=True)
    max_int8 = np.array(127, dtype=dtype)
    scale = max_int8 / abs_max
    quantized = np.round(x / abs_max * max_int8)
    # print(np.min(quantized), np.max(quantized))
    quantized = np.clip(quantized, -max_int8, max_int8).astype(dtype)
    return quantized, scale.squeeze(-1)

def block_abs_max_int8_quantize(x, block_size=32):
    x = x.reshape(*x.shape[:-1], -1, block_size)
    x, scale = abs_max_int8_quantize(x)
    # print(np.bincount((x.astype(np.int16) + 127).reshape(-1)))
    # raise ValueError
    x = x.reshape(*x.shape[:-2], -1)
    return x, scale