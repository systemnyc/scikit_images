from skimage import data, color
import matplotlib.pyplot as plt
import numpy as np

def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_rgb_image(image, title='Image'):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def import_image(image_file):
   return plt.imread(image_file)

def flipImageHorizonal(image):
   return np.fliplr(image)

def imageSize(image):
    return image.size

def imageShape(image):
    return image.shape

def tune_rgb(image, rgb=None):
    if rgb is None:
        rgb ={"r":1,"g":1, "b":1}
    tuned = image.copy()

    if rgb["r"] == 0:
        tuned[:, :, 0] = 0

    if rgb["g"] == 0:
        tuned[:, :, 1] = 0
    if rgb["b"] == 0:
        tuned[:, :, 2] = 0

    return tuned
