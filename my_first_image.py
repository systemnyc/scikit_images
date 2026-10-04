from skimage import data, color, exposure
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
    selection = {"up": np.flipud, "horizantal": np.fliplr}
    chosen = selection[direction]
    return chosen(image)


def imageSize(image):
    return image.size


def imageShape(image):
    return image.shape


def tune_rgb(image, rgb):
    return image[:, :, rgb]


def show_histogram(image, title="Title", bins=256):
    plt.hist(image.ravel(), bins)
    plt.title(title)
    plt.show()


def plot_comparison(orginal, filtered, title_filtered):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 6), sharex=True, sharey=True)

    ax1.imshow(orginal, cmap=plt.cm.gray)
    ax1.set_title("orginal")
    ax1.axis("off")
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(title_filtered)
    ax2.axis("off")
    plt.show()


def rotate_image(image, degree: int):

    original = image
    image_rotated = rotate(image, degree)
    show_image(original, "Orginal")
    rotated_by = f"""Rotated by {degree} degrees anticlockwise."""
    show_image(image_rotated, rotated_by)


def rescale_imag(image, scale, channel_axis=-1):

    original_image = image
    image_rescaled = rescale(image, scale, channel_axis=channel_axis)
    show_image(original_image, "Orginal")
    show_image(image_rescaled, "Rescaled Image")


# def resize_image (image, size:tuple[int, int]):

#     # Using tuple unpacking
#     height, width = size
#     resize_image = resize(image, (height, width))
#     show_image(resize_image, f"""Image resized""" )


def resize_image(image, size: tuple[int, int], x_smaller=None, antialias=False):
    """Resize the image proportionally"""
    height, width = size
    if x_smaller is None:
        resized_image = resize(image, (height, width), anti_aliasing=antialias)
        show_image(resized_image, "Image resized")
    else:
        height = int(image.shape[0] / x_smaller)
        width = int(image.shape[1] / x_smaller)
        resized_image = resize(image, (height, width), anti_aliasing=antialias)
        show_image(resized_image, f"""Image height = {height}, width = {width}""")
