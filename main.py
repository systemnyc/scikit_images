import numpy as np
from skimage import data, color, exposure, morphology, measure
from skimage.filters import (
    try_all_threshold,
    threshold_local,
    threshold_otsu,
    gaussian,
    sobel,
)
from skimage.restoration import inpaint, denoise_tv_chambolle, denoise_bilateral
from skimage.transform import rotate, rescale
from skimage.util import random_noise
from skimage.segmentation import slic
from skimage import io
from skimage.feature import (
    canny,
    corner_harris,
    corner_peaks,
    Cascade
)
import matplotlib.pyplot as plt
import image_utils as img


def main():

    # -------------------------
    # RGB to grayscale
    # -------------------------

    rocket = data.rocket()

    grayscale = color.rgb2gray(rocket)

    img.show_image(rocket, "Original RGB image")
    img.show_image(grayscale, "Grayscale")

    # -------------------------
    # Import and flip image
    # -------------------------

    lil_girl = img.import_image(
        r"images\little_girl.jpg"
    )

    print(type(lil_girl))
    print(lil_girl.shape)
    print(lil_girl.size)

    img.show_image(lil_girl, "Original")

    horizontally_flipped = img.flip_ops(
        lil_girl,
        "horizontal"
    )

    img.show_image(
        horizontally_flipped,
        "Horizontal Image"
    )

    # -------------------------
    # RGB channels
    # -------------------------

    red_image = lil_girl.copy()

    red_image[:, :, 1] = 0
    red_image[:, :, 2] = 0

    img.show_rgb_image(
        red_image,
        "Red Channel"
    )

    img.show_histogram(
        lil_girl,
        "Original Image"
    )

    lady_red = img.import_image(
        r"images\lady_red.jpg"
    )

    get_red_channel = img.tune_rgb(
        lady_red,
        2
    )

    img.show_histogram(
        get_red_channel,
        "Red Channel"
    )

    # -------------------------
    # Thresholding
    # -------------------------

    lady_image_gray = color.rgb2gray(
        lady_red
    )

    thresh = threshold_otsu(
        lady_image_gray
    )

    two_tone_image = (
        lady_image_gray > thresh
    )

    img.show_image(
        two_tone_image,
        "Two Tone Image"
    )

    block_size = 35

    local_threshold = threshold_local(
        lady_image_gray,
        block_size,
        offset=0.09
    )

    binary_local = (
        lady_image_gray > local_threshold
    )

    img.show_image(
        binary_local,
        "Local Thresholding"
    )

    # # -------------------------
    # # Gaussian filter
    # # -------------------------

    img3 = img.import_image(r"images\me.jpg")

    gaussian_image = gaussian(
        img3,
        channel_axis=-1
    )

    img.plot_comparison(
        img3,
        gaussian_image,
        "Blurred with love"
    )

    # -------------------------
    # Histogram equalization
    # -------------------------

    img3_gray = color.rgb2gray(img3)

    image_eq = exposure.equalize_hist(
        img3_gray
    )

    image_eq_adapt = exposure.equalize_adapthist(
        img3_gray,
        clip_limit=0.03
    )

    img.show_image(
        img3_gray,
        "Original"
    )

    img.show_image(
        image_eq,
        "Histogram Equalized"
    )

    img.show_image(
        image_eq_adapt,
        "Adaptive Histogram Equalized"
    )

    # -------------------------
    # Rotation
    # -------------------------

    rotated_image = rotate(
        img3,
        90
    )

    img.show_rgb_image(
        rotated_image,
        "Rotated 90 Degrees"
    )

    # -------------------------
    # Rescaling
    # -------------------------

    scaled_one_quarter = rescale(
        img3,
        1 / 4,
        channel_axis=-1
    )

    scaled_one_thirtieth = rescale(
        img3,
        1 / 30,
        channel_axis=-1
    )

    img.show_rgb_image(
        scaled_one_thirtieth,
        "Scaled Down"
    )

    img.plot_comparison(
        scaled_one_quarter,
        scaled_one_thirtieth,
        "Image Scale Comparison"
    )

    # -------------------------
    # Custom resize function
    # -------------------------

    resized_a = img.resize_image(
        img3,
        (1150, 900)
    )

    resized_b = img.resize_image(
        img3,
        (800, 600),
        50
    )

    img.plot_comparison(
        resized_a,
        resized_b,
        "Resized Image Comparison"
    )

    # -------------------------
    # Adaptive equalization
    # -------------------------

    original_image = data.coffee()

    adapthist_eq_image = exposure.equalize_adapthist(
        original_image,
        clip_limit=0.03
    )

    img.plot_comparison(
        original_image,
        adapthist_eq_image,
        "#ImageProcessingDatacamp"
    )

    # -------------------------
    # Morphology - Erosion
    # -------------------------

    image_horse = data.horse()

    # Horse is False, background is True.
    # Invert so horse becomes foreground.
    image_horse = np.logical_not(
        image_horse
    )

    footprint = morphology.footprint_rectangle(
        (12, 6)
    )

    eroded_image = morphology.erosion(
        image_horse,
        footprint=footprint
    )

    img.plot_comparison(
        image_horse,
        eroded_image,
        "Erosion"
    )

    # ----------------------
    # Morphology Dilatation
    # ----------------------

    dilated_image = morphology.dilation(
        image_horse
    )

    img.plot_comparison(
        image_horse,
        dilated_image,
        "Dialated Horse"
    )

    # -----------------------------
    # Image Restruction inpainting
    # -----------------------------

    defect_image = plt.imread(r"images\damaged_astronaut.png")

    mask = np.zeros(defect_image.shape[:-1])

    mask[110:250, 160:220] = 1

    restored_image = inpaint.inpaint_biharmonic(
        defect_image,
        mask,
        channel_axis = -1
    )
    img.show_image(restored_image, "This image is repaired")

    # ----------------------------
    # Image Noise
    # ----------------------------

    # Add noise to image
    noise_image = random_noise(img3)
    img.show_image(noise_image, "Image with noise")

    # TV_Chambolle
    denoised_image  = denoise_tv_chambolle(
        noise_image,
        weight=0.1,
        channel_axis=-1
    )

    img.show_image(denoised_image, "Image Denoised")

    # Apply Bilatera filter denoising

    denoised_bilateral = denoise_bilateral(noise_image, channel_axis=-1)
    img.show_image(denoised_bilateral, "Denoised Bilateral")

    # ----------------
    # Segmentation
    # ----------------

    """ 
        Unsupervised segmentation: attempt to subdivide automatically with no prior knowledge
        Simple Linear Interative Clustering (SLIC)
    """

    # Create the lables or segments
    segments = slic(img3)
    # Put lables ontop of the original image to compare
    segmented_image = color.label2rgb(segments, img3, kind='avg')

    img.show_image(img3, "Original Image")
    img.show_image(segmented_image, "Segmented Image")

    """ Achive more segmentes using n_segments """
    segmented_n_segments = slic(img3, n_segments=400)
    segmented_image = color.label2rgb(segmented_n_segments, img3, kind='avg')

    # img.show_image(segmented_n_segments, "Image with specified segments")

    # ------------------------------
    # Contures using scikit-image
    # ------------------------------

    image_2_gray = color.rgb2gray(img3)

    # img.show_image(image_2_gray, "Gray Image")

    # Make the image black and white
    thresh = threshold_otsu(image_2_gray)

    # Apply thresholding
    threasholding = image_2_gray > thresh

    # find contours
    contours = measure.find_contours(threasholding, 0.8)

    img.show_image_contours(img3, contours)

    # # ------------------------
    # # Countining Coins
    # #-------------------------

    # Get coin image
    coin_image2gray = data.coins()

    # # Apply thresholding
    coin_thresh = threshold_otsu(coin_image2gray)

    # # Get threshold
    coin_thresholding = coin_image2gray > coin_thresh

    # Detect edges 
    coin_edges = sobel(coin_image2gray)
    img.show_image(
            coin_edges,
            "Sobel edges"
        )

    edge_thresh = threshold_otsu(coin_edges)

    coin_edges_bw = coin_edges > edge_thresh

    # Find Contoursz
    get_coin_contours = measure.find_contours(
        coin_edges, 
        0.8
    )

    # Create list with the shape of each contour
    shape_contours = [
        cnt.shape[0] for cnt in get_coin_contours]

    # read your contour list
    print(shape_contours, end="\n\n")

    # associate a number with an actual contour
    for i, contour in enumerate(get_coin_contours):
        print(i, contour.shape[0])

    coin_contours = []

    for contour in get_coin_contours:

        # Number of points in contour
        points = contour.shape[0]

        # Get contour boundaries
        x_min = contour[:, 1].min()
        x_max = contour[:, 1].max()

        y_min = contour[:, 0].min()
        y_max = contour[:, 0].max()

        # Calculate width and height
        width = x_max - x_min
        height = y_max - y_min

        # Compare width and height 
        ratio = width /height

        # DEBUG: print larger contours so we can inspect them
        if points > 100:
            print(
                "points:", points,
                "width:", round(width, 1),
                "height:", round(height, 1),
                "ratio:", round(ratio, 2)
        )

        # Keep contours that look coin-like
        if(
            points > 150 
            and width > 20
            and height > 20
            and 0.8 < ratio < 1.2
        ):
            coin_contours.append(contour)
 
    img.show_image_contours(
        coin_image2gray,
        coin_contours
    )
    # # Set 50 as the maximum size of the coin shape 
    # max_coin_shape = 50

    # # Count coins in the contours excluding bigger coin size
    # coin_contours = [
    #     cnt for cnt in get_coin_contours 
    #     if np.shape(cnt)[0] < max_coin_shape]

    # # show all contours found 
    # img.show_image_contours(coin_image2gray, get_coin_contours)

    # # print the coin count 
    print("Number of coins: {}. ".format(len(coin_contours)))

    # ------------------------------------
    # Canny Edge detection from feature
    # ------------------------------------

    fruit = plt.imread(r"images\toronjas.jpg")

    # Convert to grayscale
    fruit_2gray = color.rgb2gray(fruit)

    # Apply Canny dectector
    canny_edges= canny( fruit_2gray)
    img.show_image(canny_edges, "Canny Detector!")
    canny_edge_0_5 = canny( fruit_2gray, sigma=1.8)
    img.show_image(canny_edge_0_5 , "Canny Detector!")

    # -----------------------------------------------------
    # Corner Detection
    #------------------------------------------------------

    japanese_gate = plt.imread(r"images\j_gate.jpg")
    j_graygate = color.rgb2gray(japanese_gate)
    measure_image= corner_harris(j_graygate)

    coords = corner_peaks(corner_harris(j_graygate), min_distance=15, threshold_rel=0.02)
    img.show_image(japanese_gate, "OG")
    img.show_image_with_corners(japanese_gate, coords)

    # -----------------------------------
    # Face Dectection 
    # -----------------------------------
    my_image  = plt.imread(r"images\group_image.jpg")
    
    """ 
    We are going to get the shape of the image in order to get the height and width. 
    This information will be used as the proportions in the detection function
    
    """
    height, width = my_image.shape[:2]

    min_face = int(min(height, width) * 0.05)
    max_face= int(max(height, width) * 0.60)

    print("Image shape:", my_image.shape)
    print("min_face:", min_face)
    print("max_face:", max_face)

    trained_file = data.lbp_frontal_face_cascade_filename()
    detector = Cascade(trained_file)

    # Detect faces
    detected = detector.detect_multi_scale(
        img = my_image,
        scale_factor=1.2,
        step_ratio=1,
        min_size=(min_face, min_face),
        max_size=(max_face, max_face),
        min_neighbor_number = 8
    )


    print(detected)
    img.show_detected_face(my_image, detected)

    #--------------------------------
    # Privacy Filters
    # --------------------------------

    # For each detected face
    resulting_image = my_image.copy()

    for d in detected:
        # obtain the gace cropped from the detected coordinates
        face = img.getFace(my_image, d)
        #Apply gaussian filter to extracted face
        gaussian_face = gaussian(face, channel_axis=1, sigma= 10)
        # Merge the blurry face to our final image and show it
        resulting_image = img.mergeBlurryFace(resulting_image, gaussian_face, d)

    img.show_image(resulting_image, "Blurry Image")


if __name__ == "__main__":
    main()
 