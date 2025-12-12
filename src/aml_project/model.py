import json
import math
import warnings
from pathlib import Path
from typing import Literal
import argparse
import numpy as np
import cv2

import lpips
import torch
import transformers
from performer_pytorch import Performer
from torch import nn
from tqdm import tqdm

from aml_project.dataset import ImageOnlyDataset
from aml_project.preprocess import Processor
from aml_project.view import view_images
from config import Config

BEST_WEIGHTS = Path("data/model_weights")


def sample_ellipses_mask(
    resolution: tuple[int, int],
    count_range: tuple[int, int],
    config: Config,
    sizes: tuple[float, float],
    device="cpu",
):
    """
    Returns a binary mask of specified resolution with OR of 'count' random ellipses.
    """
    h, w = resolution
    count = torch.randint(
        count_range[0],
        count_range[1] + 1,
        (),
    )
    # Create coordinate grid
    ys = torch.linspace(
        -1, 1, h, device=config.detect_device(), dtype=config.get_dtype_pt()
    )
    xs = torch.linspace(
        -1, 1, w, device=config.detect_device(), dtype=config.get_dtype_pt()
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (h, w)

    # Random ellipse parameters
    # centers in [-1..1], axes in [0.05, 0.2], angle in [0..π]
    cx = torch.empty(
        count, device=config.detect_device(), dtype=config.get_dtype_pt()
    ).uniform_(-1, 1)  # type:ignore
    cy = torch.empty(
        count, device=config.detect_device(), dtype=config.get_dtype_pt()
    ).uniform_(-1, 1)  # type:ignore
    ax = torch.empty(
        count, device=config.detect_device(), dtype=config.get_dtype_pt()
    ).uniform_(sizes[0], sizes[1])  # type:ignore
    ay = torch.empty(
        count, device=config.detect_device(), dtype=config.get_dtype_pt()
    ).uniform_(sizes[0], sizes[1])  # type:ignore
    angle = torch.empty(
        count, device=config.detect_device(), dtype=config.get_dtype_pt()
    ).uniform_(0, torch.pi)  # type:ignore

    # Expand grid to (count, h, w)
    xx = xx.unsqueeze(0)  # (1, h, w)
    yy = yy.unsqueeze(0)

    # Shift grid by ellipse center
    X = xx - cx[:, None, None]
    Y = yy - cy[:, None, None]

    # Rotate coordinates
    cos_t = angle.cos()[:, None, None]
    sin_t = angle.sin()[:, None, None]

    Xr = X * cos_t + Y * sin_t
    Yr = -X * sin_t + Y * cos_t

    # Ellipse equation (inside if <=1)
    inside = (Xr / ax[:, None, None]) ** 2 + (Yr / ay[:, None, None]) ** 2 <= 1.0

    # OR across ellipses → (h, w)
    mask = inside.any(dim=0)

    return 1 - mask.to(config.get_dtype_pt())

def sample_lines_mask(
    resolution: tuple[int, int],
    count_range: tuple[int, int],
    config: Config,
    sizes: tuple[float, float],
    device="cpu",
) -> torch.Tensor:
    """
    Returns a binary mask with random lines drawn between random points,
    with thickness defined by sizes.
    """
    H, W = resolution
    mask_np = np.ones((H, W), dtype=np.float32)

    count = torch.randint(count_range[0], count_range[1] + 1, (1,)).item()

    for _ in range(count):
        x1, y1 = np.random.randint(0, W), np.random.randint(0, H)
        x2, y2 = np.random.randint(0, W), np.random.randint(0, H)

        # Choose random thickness based on sizes (scaled by image size)
        rel_thickness = np.random.uniform(sizes[0], sizes[1])
        thickness = max(1, int(rel_thickness * min(H, W)))

        # Draw black line (0) on white mask (1)
        cv2.line(mask_np, (x1, y1), (x2, y2), color=0, thickness=thickness)

    # Convert to PyTorch tensor
    mask = torch.from_numpy(mask_np).to(device=config.detect_device(), dtype=config.get_dtype_pt())
    return mask

def mask_batch(
    batch: torch.Tensor, config: Config, device: str, steps: int, max_steps: int
):
    assert config.resolution == batch.shape[2:], "batch is of an incorrect resolution"
    progress = steps / max_steps

    progress = min(progress, config.mask_warmup_percentage) * (
        1 / config.mask_warmup_percentage
    )

    sizes = (
        config.size_range_start[0] * (1 - progress)
        + config.size_range_end[0] * progress,
        config.size_range_start[1] * (1 - progress)
        + config.size_range_end[1] * progress,
    )

    ell = [
        # sample_ellipses_mask(
        #     config.resolution,
        #     config.num_ellipses_train,
        #     config=config,
        #     device=device,
        #     sizes=sizes,
        # )
        sample_lines_mask(
	        config.resolution,
	        config.num_lines_train,
	        config=config,
	        device=device,
	        sizes=sizes,
        )
        for i in range(batch.shape[0])
    ]
    ell = torch.stack(ell, dim=0).unsqueeze(1)
    masked = ell * (batch) + (1 - ell) * torch.randn_like(batch) * 0.1
    masked = torch.cat((masked, ell), 1)
    return masked

class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device, dtype, stride, config, padding, residual=True):
        super().__init__()
        self.conv_a = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            device=device,
            dtype=dtype,
            padding=padding
        )
        self.conv_b = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            device=device,
            dtype=dtype,
            padding=padding
        )
        self.activation = config.get_activation()
        self.residual = residual
    def forward(self, x):
      x_a = self.activation(self.conv_a(x))
      x_gate = self.conv_b(x)
      result = x_a * x_gate
      if self.residual:
          result += x
      return result


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, height: int, width: int, config: Config):
        super().__init__()

        self.d_model = d_model
        self.height = height
        self.width = width

        pe = torch.zeros(
            d_model, height, width, dtype=config.get_dtype_pt()
        )  # (C, H, W)

        d_model_half = d_model // 2
        div_term = torch.exp(
            torch.arange(0, d_model_half, 2, dtype=config.get_dtype_pt())
            * (-math.log(10000.0) / d_model_half)
        )  # (d_model_half/2,)

        # Positional encoding for Y (rows)
        pos_y = torch.arange(0, height, dtype=config.get_dtype_pt()).unsqueeze(
            1
        )  # (H, 1)
        pe_y = torch.zeros(
            height, d_model_half, dtype=config.get_dtype_pt()
        )  # (H, d_model_half)
        pe_y[:, 0::2] = torch.sin(pos_y * div_term)
        pe_y[:, 1::2] = torch.cos(pos_y * div_term)

        # Positional encoding for X (cols)
        pos_x = torch.arange(0, width, dtype=config.get_dtype_pt()).unsqueeze(
            1
        )  # (W, 1)
        pe_x = torch.zeros(
            width, d_model_half, dtype=config.get_dtype_pt()
        )  # (W, d_model_half)
        pe_x[:, 0::2] = torch.sin(pos_x * div_term)
        pe_x[:, 1::2] = torch.cos(pos_x * div_term)

        # Combine Y and X to (H, W, d_model)
        pe_y = pe_y.unsqueeze(1).repeat(1, width, 1)  # (H, W, d_model_half)
        pe_x = pe_x.unsqueeze(0).repeat(height, 1, 1)  # (H, W, d_model_half)
        pe_2d = torch.cat([pe_y, pe_x], dim=-1)  # (H, W, d_model)

        # Rearrange to (1, H*W, d_model) for easy addition to sequences
        pe_2d = pe_2d.view(height * width, d_model).unsqueeze(0)  # (1, H*W, d_model)
        pe_2d = pe_2d.to(config.detect_device())

        self.register_buffer("pe", pe_2d, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H*W, d_model) where H and W match the height/width in __init__.
        """
        B, L, D = x.shape
        if D != self.d_model:
            raise ValueError(f"d_model mismatch: got {D}, expected {self.d_model}")
        if L != self.height * self.width:
            raise ValueError(
                f"Sequence length {L} != H*W ({self.height}*{self.width} = {self.height * self.width})"
            )

        return x + self.pe  # (1, L, D) will broadcast over batch #type:ignore




class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: Config,
        in_channels: int,
        out_channels: int,
        scale: Literal["up", "down"],
        depth: int,
    ):
        super().__init__()
        match scale:
            case "down":
                self.scale = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=2,
                        stride=2,
                        device=config.detect_device(),
                        dtype=config.get_dtype_pt(),
                    ),
                )
            case "up":
                self.scale = nn.Sequential(
                    nn.Upsample(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False,
                    ),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        device=config.detect_device(),
                        dtype=config.get_dtype_pt(),
                    ),
                )
            case _:
                raise ValueError(f"Unsupported scale: {scale}")
        self.transformer = None
        
        if depth >= config.transformer_depth_min:
            H, W = config.resolution
            if scale == "down":
                t_dim = in_channels
                block_H = H // (2 ** depth)
                block_W = W // (2 ** depth)
            else:  # "up": transformer is AFTER upsample, so it sees out_channels
                t_dim = out_channels
                block_H = H // (2 ** depth)
                block_W = W // (2 ** depth)
        
            self.pos_encoding = PositionalEncoding2D(
                d_model=t_dim,
                height=block_H,
                width=block_W,
                config=config,
            )
            self.transformer = Performer(
                dim=t_dim,
                dim_head=t_dim // config.heads,
                depth=config.num_layers,
                heads=config.heads,
            ).to(device=config.detect_device(), dtype=config.get_dtype_pt())
        self.activation = config.get_activation()
        self.conv = nn.Sequential()
        
        if scale == "down":
          self.conv.append(
              nn.GroupNorm(
                  8,
                  in_channels,
                  device=config.detect_device(),
                  dtype=config.get_dtype_pt(),
              )
          )
          self.conv.append(
            GatedConv(
	             in_channels=in_channels,
	             out_channels=in_channels,
	             kernel_size=3,
	             stride=1,
	             device=config.detect_device(),
	             dtype=config.get_dtype_pt(),
		          config=config,
		          padding=1,
		            )
          )
        else:
          self.conv.append(
               nn.GroupNorm(
                   8,
                   out_channels,
                   device=config.detect_device(),
                   dtype=config.get_dtype_pt(),
               )
           )
          self.conv.append(
            GatedConv(
             in_channels=out_channels,
             out_channels=out_channels,
             kernel_size=3,
             stride=1,
             device=config.detect_device(),
             dtype=config.get_dtype_pt(),
            config=config,
            padding=1,
            )
        )

    def forward(self, x: torch.Tensor, incoming_skip_connection: torch.Tensor|None = None, scale: Literal["up", "down"] = "up") -> torch.Tensor|tuple[torch.Tensor,torch.Tensor]:
      if scale == "up":
        x = self.scale(x)
        assert incoming_skip_connection is not None, "Skip connection was none in upsample"
        x = x + incoming_skip_connection
        skip_connection = x
        x = self.conv(x)
        if self.transformer is not None:
          b, c, h, w = x.shape
          x = x.view(b, c, h * w).permute(0, 2, 1)
          x = self.pos_encoding(x)
          x = self.transformer(x)
          x = x.permute(0, 2, 1).view(b, c, h, w)
        x = x + skip_connection
        return x
      else:
        skip_connection = x
        if self.transformer is not None:
            b, c, h, w = x.shape
            x = x.view(b, c, h * w).permute(0, 2, 1)
            x = self.pos_encoding(x)
            x = self.transformer(x)
            x = x.permute(0, 2, 1).view(b, c, h, w)
        x = skip_connection + x
        x = self.conv(x)
        outgoing_skip_connection = x
        x = self.scale(x)
        
        return (x,outgoing_skip_connection)

class Photosciop(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.in_projection = nn.Conv2d(
            4,
            config.get_base_dim(),
            kernel_size=1,
            device=config.detect_device(),
            dtype=config.get_dtype_pt(),
        )

        self.conv_encoder = nn.ModuleList()
        for i in range(config.conv_blocks):
            self.conv_encoder.append(
                TransformerBlock(
                    config,
                    in_channels=config.get_base_dim() * (2**i),
                    out_channels=config.get_base_dim() * (2 ** (i + 1)),
                    scale="down",
                    depth=i,
                )
            )

        H, W = config.resolution
        self.pos_encoding = PositionalEncoding2D(
            d_model=config.bottleneck,
            height=H // (2**config.conv_blocks),
            width=W // (2**config.conv_blocks),
            config=config,
        )

        self.encoder = Performer(
            dim=config.bottleneck,
            dim_head=config.bottleneck // config.heads,
            depth=config.num_layers,
            heads=config.heads,
        ).to(device=config.detect_device(),dtype=config.get_dtype_pt())

        self.conv_decoder = nn.ModuleList()
        for i in reversed(range(config.conv_blocks)):
            self.conv_decoder.append(
                TransformerBlock(
                    config,
                    in_channels=config.get_base_dim() * (2 ** (i + 1)),
                    out_channels=config.get_base_dim() * (2**i),
                    scale="up",
                    depth=i,
                )
            )

        self.out_projection = nn.Conv2d(
            config.get_base_dim(),
            3,
            kernel_size=1,
            device=config.detect_device(),
            dtype=config.get_dtype_pt(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_list: list[torch.Tensor] = []

        x = self.in_projection(x)
        output_list.append(x)
        for layer in self.conv_encoder:
            x,skip = layer(x,None,"down")
            output_list.append(skip)

 
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)

        for layer in self.conv_decoder:
            skip_connection = output_list.pop(-1)
            x = layer(x,skip_connection,"up")

        skip_connection = output_list.pop(-1)
        x += skip_connection
        return self.out_projection(x)


MODEL_PATH = "data/model_weights"


def train(
    model: nn.Module,
    train: torch.utils.data.Dataset,
    val: torch.utils.data.Dataset,
    config: Config,
    start_epoch: int = 0
) -> dict[str, list[float]]:  # type:ignore
    num_epochs = config.num_epochs
    lr = config.lr

    sampler = torch.utils.data.RandomSampler(train, replacement=False, num_samples=2500)  # type:ignore
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=config.batch_size, sampler=sampler, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    preprocessor = Processor(config)

    with warnings.catch_warnings(action="ignore"):
        lpips_loss = lpips.LPIPS(net="vgg").to(config.detect_device())

    def loss_fn(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        masked_output_image = predicted * (1 - mask) + target * mask
        return (
            predicted.sub(target).abs().mul(1 - mask).sum() / (1 - mask).sum()
            + lpips_loss(masked_output_image, target).sum() / (1 - mask).sum()
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=getattr(config, "weight_decay", 0),
    )

    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_epochs * len(train_loader),
        num_cycles=0.5,
    )

    history: list[dict[str, float]] = []

    best_loss = float("inf")
    steps = start_epoch * len(train_loader)
    max_steps = len(train_loader) * num_epochs
    for epoch in range(start_epoch,num_epochs):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in (
            pbar := tqdm(
                train_loader, desc=f"Train {epoch + 1}/{num_epochs}", colour="blue"
            )
        ):
            steps += 1
            batch = batch.to(config.detect_device())
            batch = preprocessor(batch)
            targets = batch
            inputs = mask_batch(batch, config, config.detect_device(), steps, max_steps)
            mask = inputs[:, 3:4, :, :]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": running_loss / n_batches})

        epoch_train_loss = running_loss / n_batches

        # ---- VALIDATION (optional) ----
        epoch_val_loss = None
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in (
                    pbar := tqdm(
                        val_loader,
                        desc=f"Validation {epoch + 1}/{num_epochs}",
                        colour="yellow",
                    )
                ):
                    batch = batch.to(config.detect_device())
                    batch = preprocessor(batch)
                    targets = batch
                    inputs = mask_batch(batch, config, config.detect_device(), 1, 1)
                    mask = inputs[:, 3:4, :, :]

                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets, mask)

                    val_running_loss += loss.item()
                    val_batches += 1
                    pbar.set_postfix({"loss": val_running_loss / val_batches})

            epoch_val_loss = val_running_loss / max(val_batches, 1)
            if best_loss > epoch_val_loss:
                best_loss = epoch_val_loss
                Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)
                print("New best, weights saved")
        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f}"
        )
        history.append({"val_loss": epoch_val_loss, "train_loss": epoch_train_loss})  # type:ignore
        json.dump(history, open("data/loss_history.json", "w"))


def main():
    # test
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-best", action="store_true",
                        help="Load best weights and keep training")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch number to start training from (e.g. 26)")
    args = parser.parse_args()

    config = Config()
    model = Photosciop(config)
    torch.manual_seed(config.random_seed)


    if args.resume_best:
        if not BEST_WEIGHTS.exists():
            raise FileNotFoundError(f"Best weights not found at {BEST_WEIGHTS}")
        state = torch.load(BEST_WEIGHTS, map_location=config.detect_device())
        model.load_state_dict(state)
        print(f"Loaded best weights from {BEST_WEIGHTS}")

    tra = ImageOnlyDataset(config, "train")
    val = ImageOnlyDataset(config, "validation")
    train(model, tra, val, config, start_epoch=args.start_epoch)
    pass


if __name__ == "__main__":
    main()
