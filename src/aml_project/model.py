from typing import Literal

import torch
from torch import nn

from aml_project.dataset import ImageOnlyDataset
from config import Config


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe : torch.Tensor
        # self.pe = nn.UninitializedParameter() # old ver, maybe faster
        self.current_shape = None

    def initialize_encoding(self, d_model: int, max_len: int, device: torch.device, dtype: torch.dtype):
        pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.pe = nn.Parameter(pe, requires_grad=False)
        self.current_shape = (1, max_len, d_model)
        self.register_buffer(name="pe", tensor=pe, persistent=False) # maybe slow 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.current_shape != (1, x.size(1), x.size(2)) or self.pe.device != x.device or self.pe.dtype != x.dtype:
            self.initialize_encoding(d_model=x.size(-1), max_len=x.size(1), device=x.device, dtype=x.dtype)
        x = x + self.pe[:, :x.size(1), :]
        return x

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
            # Assume batch is (inputs, targets)
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

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

    train(model, tra, val, config)
    pass


if __name__ == "__main__":
    main()
