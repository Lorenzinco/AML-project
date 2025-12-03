from typing import Literal

import torch
from torch import nn
import math 

from aml_project.dataset import ImageOnlyDataset
from config import Config

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt


def sample_ellipses_mask(resolution:tuple[int, int], count_range: tuple[int, int], device="cpu"):
    """
    Returns a binary mask of specified resolution with OR of 'count' random ellipses.
    """
    h, w = resolution
    count = torch.randint(count_range[0], count_range[1] + 1, (),)
    # Create coordinate grid
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (h, w)

    # Random ellipse parameters
    # centers in [-1..1], axes in [0.05, 0.2], angle in [0..π]
    cx = torch.empty(count, device=device).uniform_(-1, 1)
    cy = torch.empty(count, device=device).uniform_(-1, 1)
    ax = torch.empty(count, device=device).uniform_(0.05, 0.2)
    ay = torch.empty(count, device=device).uniform_(0.05, 0.2)
    angle = torch.empty(count, device=device).uniform_(0, torch.pi)

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
    inside = (Xr / ax[:, None, None])**2 + (Yr / ay[:, None, None])**2 <= 1.0

    # OR across ellipses → (h, w)
    mask = inside.any(dim=0)

    return 1 - mask.float()

class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()

        self.d_model = d_model
        self.height = height
        self.width = width

        pe = torch.zeros(d_model, height, width)  # (C, H, W)

        d_model_half = d_model // 2
        div_term = torch.exp(
            torch.arange(0, d_model_half, 2).float()
            * (-math.log(10000.0) / d_model_half)
        )  # (d_model_half/2,)

        # Positional encoding for Y (rows)
        pos_y = torch.arange(0, height, dtype=torch.float).unsqueeze(1)  # (H, 1)
        pe_y = torch.zeros(height, d_model_half)  # (H, d_model_half)
        pe_y[:, 0::2] = torch.sin(pos_y * div_term)
        pe_y[:, 1::2] = torch.cos(pos_y * div_term)

        # Positional encoding for X (cols)
        pos_x = torch.arange(0, width, dtype=torch.float).unsqueeze(1)  # (W, 1)
        pe_x = torch.zeros(width, d_model_half)  # (W, d_model_half)
        pe_x[:, 0::2] = torch.sin(pos_x * div_term)
        pe_x[:, 1::2] = torch.cos(pos_x * div_term)

        # Combine Y and X to (H, W, d_model)
        pe_y = pe_y.unsqueeze(1).repeat(1, width, 1)      # (H, W, d_model_half)
        pe_x = pe_x.unsqueeze(0).repeat(height, 1, 1)     # (H, W, d_model_half)
        pe_2d = torch.cat([pe_y, pe_x], dim=-1)           # (H, W, d_model)

        # Rearrange to (1, H*W, d_model) for easy addition to sequences
        pe_2d = pe_2d.view(height * width, d_model).unsqueeze(0)  # (1, H*W, d_model)

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

        return x + self.pe  # (1, L, D) will broadcast over batch

class ConvBlock(nn.Module):
    def __init__(
        self,
        config: Config,
        in_channels: int,
        out_channels: int,
        scale: Literal["up", "down"],
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=config.kernel_size,
            device=config.detect_device(),
            padding=config.kernel_size // 2,
            padding_mode="reflect",
        )
        self.scale = nn.Upsample(scale_factor=2) if scale == "up" else nn.MaxPool2d(2)
        self.activation = config.get_activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return self.scale(x)


class Photosciop(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.in_projection = nn.Conv2d(
            4, config.get_base_dim(), kernel_size=1, device=config.detect_device()
        )

        self.conv_encoder = nn.ModuleList()
        for i in range(config.conv_blocks):
            self.conv_encoder.append(
                ConvBlock(
                    config,
                    in_channels=config.get_base_dim() * (2**i),
                    out_channels=config.get_base_dim() * (2 ** (i + 1)),
                    scale="down",
                )
            )
            
        H, W = config.resolution
        self.pos_encoding = PositionalEncoding2D(
            d_model=config.bottleneck,
            height=H // (2 ** config.conv_blocks),
            width=W // (2 ** config.conv_blocks),
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.bottleneck,
                config.heads,
                config.dim_feed_forward,
                config.dropout,
                activation=config.activation,
                device=config.detect_device(),
                batch_first=True,
            ),
            config.num_layers,
            mask_check=False,
        )

        self.conv_decoder = nn.ModuleList()
        for i in reversed(range(config.conv_blocks)):
            self.conv_decoder.append(
                ConvBlock(
                    config,
                    in_channels=config.get_base_dim() * (2 ** (i + 2)),
                    out_channels=config.get_base_dim() * (2**i),
                    scale="up",
                )
            )

        self.out_projection = nn.Conv2d(
            config.get_base_dim() * 2, 3, kernel_size=1, device=config.detect_device()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_list: list[torch.Tensor] = []

        x = self.in_projection(x)

        for layer in self.conv_encoder:
            output_list.append(x)
            x = layer(x)

        output_list.append(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)

        for layer in self.conv_decoder:
            skip_connection = output_list.pop(-1)
            print(f"SkipConnectionsize, x: {skip_connection.shape}, {x.shape}")
            x = torch.cat([skip_connection, x], 1)
            x = layer(x)

        skip_connection = output_list.pop(-1)
        x = torch.cat([skip_connection, x], 1)
        return self.out_projection(x)


def train(
    model: nn.Module,
    train: torch.utils.data.Dataset,
    val: torch.utils.data.Dataset,
    config: Config,
) -> dict[str, list[float]]:
    device = torch.device(config.detect_device())
    model.to(device)

    num_epochs = config.num_epochs
    lr = config.lr

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=config.batch_size, shuffle=True
    )

    criterion = nn.L1Loss()  # you can swap with nn.MSELoss() if you prefer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=getattr(config, "weight_decay", 0.0),
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            targets = batch.to(device)
            ell = [sample_ellipses_mask(config.resolution, config.num_ellipses_train, device=device) for i in range(batch.shape[0])]
            ell = torch.stack(ell, dim=0).unsqueeze(1)
            print(batch.shape, ell.shape)
            inputs = (ell*(batch/256))
            inputs = torch.cat((inputs, ell), 1)
            print(inputs.shape)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(inputs[0].permute(1, 2, 0))
            axes[0].set_title("Image 1")
            axes[0].axis("off")

            axes[1].imshow(targets[0].permute(1, 2, 0)/256)
            axes[1].set_title("Image 2")
            axes[1].axis("off")

            plt.show()
            exit()  # once dataset is normalized we can just start training
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        epoch_train_loss = running_loss / max(n_batches, 1)
        history["train_loss"].append(epoch_train_loss)

        # ---- VALIDATION (optional) ----
        epoch_val_loss = None
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_running_loss += loss.item()
                    val_batches += 1

            epoch_val_loss = val_running_loss / max(val_batches, 1)
            history["val_loss"].append(epoch_val_loss)

        # Simple logging
        if epoch_val_loss is not None:
            print(
                f"[Epoch {epoch + 1}/{num_epochs}] "
                f"train_loss={epoch_train_loss:.4f} "
                f"val_loss={epoch_val_loss:.4f}"
            )
        else:
            print(f"[Epoch {epoch + 1}/{num_epochs}] train_loss={epoch_train_loss:.4f}")

    return history


def main():
    # test
    config = Config()
    model = Photosciop(config)
    # image = torch.randn((1, 4, config.resolution[0], config.resolution[1])).to(
    #     config.detect_device()
    # )
    # out = model(image)
    # print(f"Outshape,inShape: {out.shape} , {image.shape}")
    # assert out.shape[2:] == image.shape[2:]
    tra = ImageOnlyDataset(config, "train")
    val = ImageOnlyDataset(config, "validation")
    # TODO: normalize dataset
    train(model, tra, val, config)
    pass


if __name__ == "__main__":
    main()
