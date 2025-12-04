from matplotlib import pyplot as plt
import torch

def view_images(images:torch.Tensor | list[torch.Tensor], labels:None | list[str]=None):
    """
    Plot many images
    
    :param images: Images as float tensors with [b, c, h, w] or [c, h, w] dimensions. Also list of tensor images works
    :type images: Torch.Tensor
    :param labels: Optional labels for the plots, defaults to empty
    """
    if not isinstance(images, list):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        images = torch.unbind(images)
    n_images = len(images)
    if labels is None:
        labels = [""]*n_images

    fig, axes = plt.subplots(1, n_images, figsize=(10, 5))

    for i, image in enumerate(images):
        axes[i].imshow(image.permute(1, 2, 0).to(dtype=float))
        axes[i].set_title(labels[i])
        axes[i].axis("off")

    plt.show()

