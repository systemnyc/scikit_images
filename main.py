import numpy as np

from skimage import data, exposure, morphology
from skimage.transform import rescale

import image_utils as img

def main():

    # # # Images from datae
    # # rocket = data.rocket()
    # # # Convert rgb to grayscale
    # # grayscale = color.rgb2gray(rocket)
    # # # Show orginal Image
    # # show_image(rocket, 'Original RGB image')
    # # # Show Grayscale image
    # # show_image(grayscale, "Grayscale")

    # # Import a image and get its type
    # # lil_girl = plt.imread(r"images\little_girl.jpg")
    #     lil_girl = img.import_image(r"images\little_girl.jpg")
    #     # print(type(lil_girl))
    #     # print(lil_girl.shape)
    #     # print(lil_girl.size)
    #     # Flip the image in the left direction
    #     img.show_image(lil_girl, "Original")
    #     # horizontally_flipped = flipImageHorizonal(lil_girl)
    #     #show_image(horizontally_flipped, "Ho-flipped")
    #     img.show_image(img.flip_ops(lil_girl, "horizantal"), "Horizontial Image")
    #     red_image = img.tune_rgb(lil_girl, 0)
    #     # red_image = lil_girl.copy()
    #     # red_image[:, :,1] = 0
    #     # red_image[:, :,2] = 0
    #     # plt.imshow(red)sciki
    #     # plt.show()
    #     img.show_rgb_image(red_image, "Red Channel")
    #     img.show_histogram(lil_girl, "Original Image")

    #     lady_red = img.import_image(r"images/lady_red.jpg")
    #     get_red_channel = img.tune_rgb(lady_red, 2)
    #     img.show_histogram(get_red_channel,"red channel")

    #     # Thresholding
    #     lady_image_gray = img.color.rgb2gray(lady_red)

    #     thresh = img.threshold_otsu(lady_image_gray)
    #     two_tone_image = lady_image_gray > thresh
    #     img.show_image(two_tone_image, " Two Tone Image")

    #     block_size = 15
    #     local_threshold = img.threshold_local(lady_image_gray, block_size, offset=10)
    #     binary_local = lady_image_gray  > local_threshold
    #     img.show_image(binary_local, "Local Thresholding")

    img3 = img.import_image(r"images\PXL_20220112_200237532.jpg")
    # # Apply edge dectection filter
    # gaussian_image = img.gaussian(img3,channel_axis=2)
    # img.plot_comparison(img3, gaussian_image, "Blurred with love")

    # img3_gray = img.color.rgb2gray(img3)
    # # obtain the equalized image
    # image_eq = img.exposure.equalize_hist(img3_gray)
    # image_eq_adapt = img.exposure.equalize_adapthist(img3_gray, clip_limit=0.03)
    # # Show the orginal image
    # img.show_image(img3_gray, 'original')
    # img.show_image(image_eq,"Histogram equalized")
    # img.show_image(image_eq_adapt, "Hist Adapthist")

    # img.rotate_image(img3, 90)
    # img.rescale_imag(img3, 1/4)print(type(img3))
    scaled_one_quarter = img.rescale(img3, 1 / 4, channel_axis=-1)
    # img.show_image(img.rescale(img3, 1 / 4, channel_axis=-1), "Really Charles!")

    scaled_one_thirtith = img.rescale(img3, 1 / 30, channel_axis=-1)
    # img.show_rgb_image(scaled, "Scaled down down!")

    img.plot_comparison(
        scaled_one_quarter, scaled_one_thirtith, "Image scale comparision"
    )

    # Resize image
    a = img.resize_image(img3, (1150, 900))
    b = img.resize_image(img3, (800, 600), 50)
    img.plot_comparison(a, b, "Resized with dimension")

    # Applying Adaptive qualizaiton to Coffee image

    # Load coffee image
    original_image = img.data.coffee()

    # Apply the adaptive equalization on the original image
    adapthist_eq_image = img.exposure.equalize_adapthist(
        original_image, clip_limit=0.03
    )

    # Compare the original image to the equalized
    img.plot_comparison(original_image, adapthist_eq_image, "#ImageProcessingDatacamp")

    # Image Processing Morphology
    # Imageg distorted: Try to remove imperfection, account for form and strucutue in image
    # Dilated: add to image(Pixels)
    # Errousion remove from image
    # Applying erosion: binary_erosion function
    footprint = img.morphology.footprint_rectangle((12, 6))
    image_horse = img.data.horse()
    image_horse = img.np.logical_not(image_horse)
    erouded_image = img.morphology.erosion(image_horse, footprint=footprint)
    print(type(image_horse))
    print(image_horse.dtype)
    print(img.np.unique(image_horse))
    print(image_horse[0, 0])
    img.plot_comparison(image_horse, erouded_image, "Erosion")

    
if __name__ == "__main__":
    main()
