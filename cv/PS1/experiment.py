import numpy as np
import cv2

from ps1 import *


def main():

    # TODO: Insert your image file paths here:
    img1_filename = None
    img2_filename = None

    # # 1a
    img1 = cv2.imread(img1_filename)
    img2 = cv2.imread(img2_filename)

    assert 100 < img1.shape[0] <= 512, "Check your image 1 dimensions"
    assert 100 < img1.shape[1] <= 512, "Check your image 1 dimensions"
    assert 100 < img2.shape[0] <= 512, "Check your image 2 dimensions"
    assert 100 < img2.shape[1] <= 512, "Check your image 2 dimensions"
    assert img1.shape[1] > img1.shape[0], "Image 1 should be a wide image"
    assert img2.shape[0] > img2.shape[1], "Image 2 should be a tall image"

    cv2.imwrite('ps1-1-a-1.png', img1)
    cv2.imwrite('ps1-1-a-2.png', img2)

    # # 2 Color Planes

    # # 2a
    swapped_green_blue_img = swap_green_blue(img1)
    cv2.imwrite('ps1-2-a-1.png', swapped_green_blue_img)

    # # 2b
    img1_green = extract_green(img1)
    assert len(img1_green.shape) == 2, "The monochrome image must be a 2D array"
    cv2.imwrite('ps1-2-b-1.png', img1_green)

    # # 2c
    img1_red = extract_red(img1)
    assert len(img1_red.shape) == 2, "The monochrome image must be a 2D array"
    cv2.imwrite('ps1-2-c-1.png', img1_red)

    # # 3 Replacement of Pixels

    # # 3a

    # TODO: Choose the monochrome image for img1.
    mono1 = None

    mono2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    replaced_img = copy_paste_middle(mono1, mono2, (100, 100))

    cv2.imwrite('ps1-3-a-1.png', replaced_img)


    # # 3b
    replaced_img_circle = copy_paste_middle_circle(mono1, mono2, 50)

    cv2.imwrite('ps1-3-b-1.png', replaced_img_circle)

    # # 4 Arithmetic and Geometric operations

    # # 4a
    min_green, max_green, mean_green, stddev_green = image_stats(img1_green)

    print("The min pixel value of img1_green is", min_green)
    print("The max pixel value of img1_green is", max_green)
    print("The mean pixel value of img1_green is", mean_green)
    print("The std dev of img1_green is", stddev_green)

    # # 4b
    normalized_img = center_and_normalize(img1_green, 10)
    cv2.imwrite('ps1-4-b-1.png', normalized_img)

    # # 4c
    shift_green = shift_image_left(img1_green, 2)
    cv2.imwrite('ps1-4-c-1.png', shift_green)

    # # 4d
    diff_green = difference_image(img1_green, shift_green)
    cv2.imwrite('ps1-4-d-1.png', diff_green)

    # # 5 Noise

    # TODO: Choose a sigma value:
    sigma = None

    # # 5a
    channel = 1
    noisy_green = add_noise(img1, channel, sigma)
    cv2.imwrite('ps1-5-a-1.png', noisy_green)

    # # 5b
    channel = 0
    noisy_blue = add_noise(img1, channel, sigma)
    cv2.imwrite('ps1-5-b-1.png', noisy_blue)

    # # 7 Hybrid Images
    img1 = cv2.imread('dog.bmp')
    img2 = cv2.imread('cat.bmp')

    cutoff_frequency = 7

    hybrid_image = build_hybrid_image(img1, img2, cutoff_frequency)
    
    # Note that some processing in the following lines may be relative to your implementation.
    # If you have/have not altered the channel conventions (BGR -> RGB) and the multiplying factor of 255, you may choose to modify
    # or remove the lines 'vis.copy()*255' and 'vis[:,:,::-1] below.

    
    vis = vis_hybrid_image(hybrid_image)
    vis = (vis.copy() * 255)
    vis = vis.astype(np.uint8)
    cv2.imwrite("hybrid_image_scales.png", vis[:, :, ::-1])



if __name__ == "__main__":
    HONOUR_PLEDGE = "By submitting to gradescope, you accept that you have neither received or given aid in the assignment. Please refer to \"Honor_Code_Policy.pdf\" to know about the policy about plagiarism for this class"
    print(HONOUR_PLEDGE)

    LATE_SUBMISSION_POLICY = "I have read the late assignments policy for CS6476. I understand that only my last commit before the deadline will be accepted without penalty."
    print(LATE_SUBMISSION_POLICY)

    main()
