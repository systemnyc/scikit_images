from skimage import data, color, exposure, morphology
from skimage.filters import (
    try_all_threshold,
    threshold_local,
    threshold_otsu,
    gaussian,
    sobel,
)
from skimage.transform import rotate, rescale, resize
# import matplotlib.pyplot as plt
from  matplotlib import pyplot as plt, patches
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


def get_rgb_channels(image):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    return red_channel, green_channel, blue_channel


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


def show_image_contours(original_image, contours):

    plt.imshow(original_image, cmap="gray")

    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0])

    plt.axis("off")
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


def get_contour(image, contour, num_of_contour):

    test_contour = contour[num_of_contour]

    plt.imshow(image, cmap="gray")

    plt.plot(test_contour[:, 1], test_contour[:, 0])

    plt.show()


def show_image_with_corners(image, coords, title="Corners detected"):

    plt.imshow(image, interpolation="nearest", cmap="gray")
    plt.title(title)
    plt.plot(coords[:, 1], coords[:, 0], "+r", markersize=15)
    plt.axis("off")
    plt.show()

def show_detected_face(result, detected, title="face image"):
    
    plt.imshow(result)
    img_desc = plt.gca()
    plt.set_cmap('gray')
    plt.title(title)
    plt.axis('off')

    for patch in detected:
        img_desc.add_patch(
            patches.Rectangle(
                (patch['c'], patch['r']),
                patch['width'],
                patch['height'],
                fill=False, color='r',linewidth=2)
            )
    plt.show()


def getFace(image, d):
    """Extracts the face rectangle from the image using the coordinates of the detected."""

    x, y = d['r'], d['c']

    width = d['r'] + d['width']
    height = d['c'] + d['height']

    face = image[x:width, y:height]

    return face


def mergeBlurryFace(original, gaussian_image, d):

    x, y = d['r'], d['c']

    width = d['r'] + d['width']
    height = d['c'] + d['height']

    original[x:width, y:height] = gaussian_image

    return original