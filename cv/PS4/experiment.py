"""Problem Set 4: Motion Detection"""

import os

import cv2
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "./output_images/"

# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):

            cv2.line(img_out, (x, y),
                     (x + int(u[y, x] * scale), y + int(v[y, x] * scale)),
                     color, 1)
            cv2.circle(img_out,
                       (x + int(u[y, x] * scale), y + int(v[y, x] * scale)), 1,
                       color, 1)
    return img_out


# Functions you need to complete:
def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """
    for i in range(level):
        # expand by 2
        uuu = 2.0 * ps4.expand_image(u)
        vvv = 2.0 * ps4.expand_image(v)
        target_hhh, target_www = pyr[level - 1 - i].shape
        uuu, vvv = targ_check(uuu, vvv, target_hhh, target_www)
        u, v = uuu, vvv
    # bug woth var names
    uuu, vvv = u, v
    target_hhh, target_www = pyr[0].shape
    uuu, vvv = targ_check(uuu, vvv, target_hhh, target_www)
    return uuu, vvv


def targ_check(uuu, vvv, target_hhh, target_www):
    if uuu.shape[0] != target_hhh or uuu.shape[1] != target_www:
        uuu = uuu[:target_hhh, :target_www].copy()
        vvv = vvv[:target_hhh, :target_www].copy()
    return uuu, vvv


def part_1a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR2.png'),
                          0) / 255.
    shift_r5_u5 = cv2.imread(
        os.path.join(input_dir, 'TestSeq', 'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 22
    k_type = "uniform"
    sigma = 1.2
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    k_size = 22
    k_type = "uniform"
    sigma = 1.2
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
                           0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'),
                           0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'),
                           0) / 255.

    k_size = 22
    k_type = "uniform"
    sigma = 1.1
    # were using interp cubic and reflect101 for border mode from cv
    interpolation = cv2.INTER_CUBIC
    border_mode = cv2.BORDER_REFLECT101
    levels = 4
    tuples = [("shift_r10", shift_r10, "ps4-1-b-1.png"), ("shift_r20", shift_r20, "ps4-1-b-2.png"), ("shift_r40", shift_r40, "ps4-1-b-3.png")]
    
    for shiftname, shiftimg, outname in tuples:
        # hier lk
        # start with r0 and go to each
        uuu, vvv = ps4.hierarchical_lk(shift_0, shiftimg, levels, k_size, k_type, sigma, interpolation, border_mode)
        u_v = quiver(uuu, vvv, scale=3, stride=10)
        cv2.imwrite(os.path.join(output_dir, outname), u_v)


def part_2():

    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 4  # Define the number of pyramid levels (must be > level_id)
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 3  # valid index in [0, levels-1]
    k_size = 22
    k_type = "uniform"
    sigma = 1.2
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id], k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 4
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 3
    k_size = 22
    k_type = "uniform"
    sigma = 1.2
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id], k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'Shift0.png'),
                         0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'),
                           0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR20.png'),
                           0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq', 'ShiftR40.png'),
                           0) / 255.

    levels = 4
    k_size = 22
    k_type = "uniform"
    sigma = 1.2
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(os.path.join(input_dir, 'Urban2', 'urban01.png'),
                              0) / 255.
    urban_img_02 = cv2.imread(os.path.join(input_dir, 'Urban2', 'urban02.png'),
                              0) / 255.

    levels = 4
    k_size = 22
    k_type = "uniform"
    sigma = 1.2
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    # grab frames
    frame_0 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    frame_1 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 4
    k_size = 22
    k_type = "uniform"
    sigma = 1.2
    # interp cubic and reflect101 from cv
    interpolation = cv2.INTER_CUBIC
    border_mode = cv2.BORDER_REFLECT101
    uuu, vvv = ps4.hierarchical_lk(frame_0, frame_1, levels, k_size, k_type, sigma, interpolation, border_mode)
    ttt = 0.5
    # use warp func
    frame1fwd = ps4.warp(frame_1, ttt * uuu, ttt * vvv, interpolation, border_mode)
    # blend frames
    frame_t = (1 - ttt) * frame_0 + ttt * frame1fwd
    # ;ets see wht we got
    cv2.imwrite(os.path.join(output_dir, "ps4-5-a-1.png"), ps4.normalize_and_scale(frame_t))



def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    # grab frames
    frame_0 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    frame_7 = cv2.imread(os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 4
    k_size = 22
    k_type = "uniform"
    sigma = 1.1
    # interp cubic and reflect101 from cv
    interpolation = cv2.INTER_CUBIC
    border_mode = cv2.BORDER_REFLECT101
    uuu, vvv = ps4.hierarchical_lk(frame_0, frame_7, levels, k_size, k_type, sigma, interpolation, border_mode)

    for iii, ttt in enumerate([0.25, 0.5, 0.75]):
        frame_forward = ps4.warp(frame_7, ttt * uuu, ttt * vvv, interpolation, border_mode)
        frame_t = (1 - ttt) * frame_0 + ttt * frame_forward
        cv2.imwrite(os.path.join(output_dir, "ps4-5-b-{}.png".format(iii + 1)), ps4.normalize_and_scale(frame_t))


def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    # Video filenames
    video_names = [
        "person01_running_d2_uncomp.avi",
        "person08_running_d4_uncomp.avi",
        "person04_walking_d4_uncomp.avi",
        "person06_walking_d1_uncomp.avi",
        "person14_handclapping_d3_uncomp.avi",
        "person10_handclapping_d4_uncomp.avi",
    ]
    class_names = {1: "Running", 2: "Walking", 3: "Clapping"}
    results = []

    video_subdir = os.path.join(input_dir, "videos/")
    for video_name in video_names:
        # file pathright?
        path = os.path.join(video_subdir, video_name)
        if not os.path.isfile(path):
            print("Part 6: skip (file not found): {}".format(path))
            continue

        # frames read right>
        frames = ps4.read_video(path)
        if not frames:
            print("Part 6: no frames read for {}".format(video_name))
            continue

        # classify_video expects grayscale float in [0, 1]
        frames_gray = []
        for fff in frames:
            if len(fff.shape) == 3:
                ggg = cv2.cvtColor(fff, cv2.COLOR_BGR2GRAY)
            else:
                ggg = fff
            frames_gray.append(np.asarray(ggg, dtype=np.float64) / 255.0)

        # try out our fumction  
        label = ps4.classify_video(frames_gray)
        line = "{} -> class {} ({})".format(video_name, label, class_names[label])
        results.append(line)
        print("Part 6: {}".format(line))

    if results:
        with open(os.path.join(output_dir, "ps4-6-results.txt"), "w") as f:
            f.write("\n".join(results))


if __name__ == '__main__':
    part_1a()
    part_1b()
    part_2()
    part_3a_1()
    part_3a_2()
    part_4a()
    part_4b()
    part_5a()
    part_5b()
    part_6()
