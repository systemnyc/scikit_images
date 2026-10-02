from my_first_image import (
     show_image, 
     show_rgb_image, 
     import_image,
     flipImageHorizonal,
     tune_rgb)


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
    lil_girl = import_image(r"images\little_girl.jpg")
    # print(type(lil_girl))
    # print(lil_girl.shape)
    # print(lil_girl.size)
    # Flip the image in the left direction
    show_image(lil_girl, "Original")
    horizontally_flipped = flipImageHorizonal(lil_girl)
    show_image(horizontally_flipped, "Ho-flipped")
    red_image = tune_rgb(lil_girl, {"r": 1, "g": 0, "b": 0})
    # red_image = lil_girl.copy()
    # red_image[:, :,1] = 0
    # red_image[:, :,2] = 0
    # plt.imshow(red)
    # plt.show()
    show_rgb_image(red_image, "Red Channel")

if __name__ == "__main__":
    main()