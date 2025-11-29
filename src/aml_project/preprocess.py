import numpy as np
import torch

from config import Config


class Processor:
    def __init__(self, config: Config):
        super().__init__()
        self.channel_variances = [
            0.07534193008541379,
            0.07263886399368141,
            0.08090809892670492,
        ]
        self.channel_means = [
            0.4758321930606835,
            0.4500513839713287,
            0.40980690006523174,
        ]
        self.channel_stds = np.sqrt(self.channel_variances)
        self.dtype = config.get_dtype_pt()

        self.resolution = config.resolution

    def __call__(self, images: list[np.ndarray] | np.ndarray):
        if isinstance(images, np.ndarray):
            images = [images]
        processed_images = []
        for img in images:
            # central crop
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            h, w = img.shape[1], img.shape[2]
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            img = img[:, top : top + min_dim, left : left + min_dim]
            # resize to target resolution
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=self.resolution,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            img = img / 255.0
            for c in range(3):
                img[c, :, :] = (
                    img[c, :, :] - self.channel_means[c]
                ) / self.channel_stds[c]
            img = img.to(dtype=self.dtype)
            processed_images.append(img)
        return {"pixel_values": torch.stack(processed_images)}

    def decode(self, pixel_values: torch.Tensor):
        decoded_images = []
        for img in pixel_values:
            img = img.permute(1, 2, 0)  # CHW to HWC
            img = img.cpu().float().numpy()
            for c in range(3):
                img[:, :, c] = (
                    img[:, :, c] * self.channel_stds[c]
                ) + self.channel_means[c]
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            decoded_images.append(img)
        return np.stack(decoded_images)
