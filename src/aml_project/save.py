import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
import torch

def save_images(
    images: torch.Tensor | list[torch.Tensor],
    labels: None | list[str] = None,
    path: str = "images.png",
    dpi: int = 100,
):
    """
    Save many images as a single matplotlib figure.

    :param images: Images as float tensors with [b, c, h, w] or [c, h, w] dimensions.
                   Also list of tensor images works.
    :param labels: Optional labels for the plots, defaults to empty strings.
    :param path:   Output file path.
    :param dpi:    Dots per inch for the saved figure.
    """
    # Normalize input to list[Tensor] with shape [c, h, w]
    if not isinstance(images, list):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        images = torch.unbind(images)

    n_images = len(images)

    # Normalize labels length
    if labels is None:
        labels = [""] * n_images
    else:
        labels = list(labels)
        if len(labels) < n_images:
            labels += [""] * (n_images - len(labels))
        elif len(labels) > n_images:
            labels = labels[:n_images]

    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))

    if n_images > 1:
        for i, image in enumerate(images):
            axes[i].imshow(image.permute(1, 2, 0).to(dtype=torch.float32))  # type: ignore
            axes[i].set_title(labels[i])
            axes[i].axis("off")
    else:
        image = images[0]
        axes.imshow(image.permute(1, 2, 0).to(dtype=torch.float32))  # type: ignore
        axes.set_title(labels[0])
        axes.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

