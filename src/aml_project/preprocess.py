import numpy as np
import torch
from torchvision.transforms.functional import resize
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

    def __call__(self, images: torch.Tensor):
        processed_images = []
        for img in images:
            img = img / 255.0
            for c in range(3):
                img[c, :, :] = (
                    img[c, :, :] - self.channel_means[c]
                ) / self.channel_stds[c]
            img = img.to(dtype=self.dtype)
            processed_images.append(img)
        result = torch.stack(processed_images)
        # print(f"preprocessor output:{result.shape}")
        return result

    def denorm(self, pixel_values: torch.Tensor):
        decoded_images = []
        for img in pixel_values:
            img = img.clone().cpu().float()
            for c in range(3):
                img[c, :, :] = (
                    img[c, :, :] * self.channel_stds[c]
                ) + self.channel_means[c]
            decoded_images.append(img)
        return decoded_images

    def decode(self, pixel_values: torch.Tensor):
        decoded_images = []
        for img in pixel_values:
            img = img.cpu().float()
            for c in range(3):
                img[c, :, :] = (
                    img[c, :, :] * self.channel_stds[c]
                ) + self.channel_means[c]
            img = torch.clip(img * 255.0, 0, 255).to(dtype=torch.uint8)
            decoded_images.append(img)
        return decoded_images
