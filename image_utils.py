from skimage import data, color, exposure, morphology
from skimage.filters import (
    try_all_threshold,
    threshold_local,
    threshold_otsu,
    gaussian,
    sobel,
)
from skimage.transform import rotate, rescale, resize
import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title="Image", cmap_type="gray"):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_rgb_image(image, title="Image"):
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def import_image(image_file):
    return plt.imread(image_file)


def flip_ops(image, direction=None):
    selection = {"up": np.flipud, "horizontal": np.fliplr}
    chosen = selection[direction]
    return chosen(image)


def image_size(image):
    return image.size


def image_shape(image):
    return image.shape


def tune_rgb(image, rgb):
    return image[:, :, rgb]


def show_histogram(image, title="Title", bins=256):
    plt.hist(image.ravel(), bins)
    plt.title(title)
    plt.show()


def plot_comparison(original_image, filtered_image, title_filtered):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 6), sharex=True, sharey=True)

    ax1.imshow(original_image, cmap=plt.cm.gray)
    ax1.set_title("original")
    ax1.axis("off")
    ax2.imshow(filtered_image, cmap=plt.cm.gray)
    ax2.set_title(title_filtered)
    ax2.axis("off")
    plt.show()


def rotate_image(image, degree: int):
    return rotate(image, degree)

def rescale_image(image, scale, channel_axis=-1):
    return rescale(image, scale, channel_axis=channel_axis)


def resize_image(image, size: tuple[int, int], x_smaller=None, antialias=False):
    """Resize the image proportionally"""
    height, width = size
    if x_smaller is None:
        resized_image = resize(image, (height, width), anti_aliasing=antialias)
    else:
        height = int(image.shape[0] / x_smaller)
        width = int(image.shape[1] / x_smaller)
        resized_image = resize(image, (height, width), anti_aliasing=antialias)

    return resized_image
