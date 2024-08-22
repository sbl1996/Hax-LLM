from typing import Tuple, Literal, Optional
from flax import struct


@struct.dataclass
class RematScanConfigMixin:
    remat: bool = False
    scan: bool = False
    remat_scan: bool = False
    lengths: Tuple[int, int] = (-1, 1)

    def remat_scan_lengths(self):
        if not self.remat_scan:
            raise ValueError("remat_scan_lengths called when remat_scan is False")
        return self.lengths

    def scan_lengths(self):
        if not self.scan:
            raise ValueError("scan_lengths called when scan is False")
        return self.lengths


@struct.dataclass
class RoPEScalingConfig:
    rope_type: Literal["default", "llama3", "chatglm2", "dynamic"] = "default"
    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    max_position_embeddings: int = 8192


@struct.dataclass
class RoPEScalingConfigMixin:
    rope_scaling: Optional[RoPEScalingConfig] = None