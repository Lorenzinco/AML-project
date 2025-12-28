import numpy as np
import torch
from pydantic import BaseModel


class Config(BaseModel):
    backbone: str = "facebook/dinov2-small"
    device: str | None = None
    resolution: tuple[int, int] = (512, 512)
    dtype: str = "bfloat16"
    kernel_size: int = 3
    bottleneck: int = 256
    conv_blocks: int = 3
    in_channels: int = 4
    heads: int = 8
<<<<<<< HEAD
    dim_feed_forward: int = 256
    transformer_depth_min: int = 0
    num_layers: int = 6
    dropout: float = 0.0
    activation: str = "gelu"
    lr: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 2
    num_ellipses_train: tuple[int, int] = (3, 10)
    num_lines_train: tuple[int,int] = (6, 15) 
=======
    dim_feed_forward: int = 512
    num_layers: int = 5
    dropout: float = 0.1
    activation: str = "gelu"
    lr: float = 10e-3
    num_epochs: int = 100
    batch_size: int = 64
    num_ellipses_train: tuple[int, int] = (1, 5)
>>>>>>> main
    random_seed: int = 0
    size_range_start: tuple[float, float] = (0.005, 0.01)
    size_range_end: tuple[float, float] = (0.05, 0.1)
    mask_warmup_percentage: float = 0.1

    def get_activation(self) -> torch.nn.Module:
        activations = {
            "relu": torch.nn.ReLU(),
            "gelu": torch.nn.GELU(),
            "tanh": torch.nn.Tanh(),
        }
        return activations[self.activation]

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
