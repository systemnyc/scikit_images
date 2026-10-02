# from my_first_image import (
#      show_image, 
#      show_rgb_image, 
#      import_image,
#      tune_rgb,
#      flip_ops,
#      show_histogram)
import my_first_image as img

def main():
    
# # Images from datae
# rocket = data.rocket()
# # Convert rgb to grayscale
# grayscale = color.rgb2gray(rocket)
# # Show orginal Image 
# show_image(rocket, 'Original RGB image')
# # Show Grayscale image
# show_image(grayscale, "Grayscale")

# Import a image and get its type
# lil_girl = plt.imread(r"images\little_girl.jpg")
    lil_girl = img.import_image(r"images\little_girl.jpg")
    # print(type(lil_girl))
    # print(lil_girl.shape)
    # print(lil_girl.size)
    # Flip the image in the left direction
    img.show_image(lil_girl, "Original")
    # horizontally_flipped = flipImageHorizonal(lil_girl)
    #show_image(horizontally_flipped, "Ho-flipped")
    img.show_image(img.flip_ops(lil_girl, "horizantal"), "Horizontial Image")
    red_image = img.tune_rgb(lil_girl, 0)
    # red_image = lil_girl.copy()
    # red_image[:, :,1] = 0
    # red_image[:, :,2] = 0
    # plt.imshow(red)
    # plt.show()
    img.show_rgb_image(red_image, "Red Channel")
    img.show_histogram(lil_girl, "Original Image")

    lady_red = img.import_image(r"images/lady_red.jpg")
    get_red_channel = img.tune_rgb(lady_red, 2)
    img.show_histogram(get_red_channel,"red channel")
    
if __name__ == "__main__":
    main()