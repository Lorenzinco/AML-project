import numpy as np
import torch
from pydantic import BaseModel


class Config(BaseModel):
    backbone: str = "facebook/dinov2-small"
    device: str | None = None
    resolution: tuple[int, int] = (128, 128)
    dtype: str = "bfloat16"
    kernel_size: int = 3
    bottleneck: int = 128
    conv_blocks: int = 3
    in_channels: int = 4
    heads: int = 8
    dim_feed_forward: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "gelu"
    lr: float = 10e-3
    num_epochs: int = 100
    batch_size: int = 64

    def get_dtype_pt(self) -> torch.dtype:
        if self.dtype == "float32":
            return torch.float32
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        raise ValueError(f"Unsupported dtype: {self.dtype}")

    def get_dtype_np(self) -> type:
        if self.dtype == "float32":
            return np.float32
        if self.dtype == "float16":
            return np.float16
        if self.dtype == "bfloat16":
            return np.float32  # NumPy does not have bfloat16, use float32 as fallback
        raise ValueError(f"Unsupported dtype: {self.dtype}")

    def get_base_dim(self):
        return self.bottleneck // (2**self.conv_blocks)

    def detect_device(self):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.version.hip is not None and torch.cuda.is_available():
                return "rocm"
        except AttributeError:
            pass
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
        except Exception:
            pass
        return "cpu"
