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