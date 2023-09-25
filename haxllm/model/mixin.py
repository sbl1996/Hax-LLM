from typing import Tuple
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
