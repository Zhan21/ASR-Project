import io

import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        # channels must be in the last dim
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")  # we do not need axis
    # To create a tensor from matplotlib,
    # we need a buffer to save the figure
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image


def plot_spectrogram(spectrogram):
    """
    Plot spectrograms

    Args:
        spectrogram (Tensor): spectrogram tensor.
    Returns:
        image (Image): image of the spectrogram
    """
    num_specs = spectrogram.size(0)

    fig, axes = plt.subplots(num_specs, 1, figsize=(10, num_specs * 2))
    for i in range(num_specs):
        axes[i].pcolormesh(spectrogram[i])
        axes[i].set_ylabel(f"{i+1}   ", rotation=0)
        axes[i].get_xaxis().set_visible(False)
        axes[i].tick_params(left=False, labelleft=False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image
