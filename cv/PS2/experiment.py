import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

import ps2

INPUT_DIR = "input_images/"
OUTPUT_DIR = "output_images/"

# -- display utils
marker_color = (255, 0, 255)
text_color = (90, 90, 90)
text_thickness = 2
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5


def place_text(text, center, img, cache={}):
    if "y_offset" in cache:
        cache["y_offset"] *= -1
    else:
        cache["y_offset"] = -30
    size = cv2.getTextSize(text, font_face, font_scale, text_thickness)
    y = center[1] + cache["y_offset"]
    if size[0][0] + center[0] > img.shape[1]:
        x = center[0] - size[0][0] - 5
    else:
        x = center[0] + 5
    cv2.rectangle(img, (x, y - size[0][1] - size[1]),
                  (x + size[0][0], y + size[0][1] - size[1]), (255, 255, 255),
                  cv2.FILLED)
    cv2.putText(img, text, (x, y), font_face, font_scale, text_color,
                text_thickness)


def draw_tl_center(image_in, center, state):
    """Marks the center of a traffic light image and adds coordinates
    with the state of the current image
    Use OpenCV drawing functions to place a marker that represents the
    traffic light center. Additionally, place text using OpenCV tools
    that show the numerical and string values of the traffic light
    center and state. Use the following format:
        ((x-coordinate, y-coordinate), 'color')
    See OpenCV's drawing functions:
    http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
    Make sure the font size is large enough so that the text in the
    output image is legible.
    Args:
        image_in (numpy.array): input image.
        center (tuple): center numeric values.
        state (str): traffic light state values can be: 'red',
                     'yellow', 'green'.
    Returns:
        numpy.array: output image showing a marker representing the
        traffic light center and text that presents the numerical
        coordinates with the traffic light state.
    """
    center = (int(center[0]), int(center[1]))
    output = image_in
    cv2.drawMarker(output,
                   center,
                   marker_color,
                   markerType=cv2.MARKER_CROSS,
                   markerSize=11,
                   thickness=2)
    text = "(({}, {}), '{}')".format(center[0], center[1], state)
    place_text(text, center, output)
    return output


def mark_traffic_signs(image_in, signs_dict):
    """Marks the center of a traffic sign and adds its coordinates.
    This function uses a dictionary that follows the following
    structure:
    {'sign_name_1': (x, y), 'sign_name_2': (x, y), etc.}
    Where 'sign_name' can be: 'stop', 'no_entry', 'yield',
    'construction', and 'warning'.
    Use cv2.putText to place the coordinate values in the output
    image.
    Args:
        signs_dict (dict): dictionary containing the coordinates of
        each sign found in a scene.
    Returns:
        numpy.array: output image showing markers on each traffic
        sign.
    """
    output = image_in
    items = []
    for k, center in signs_dict.items():
        items.append((int(center[0]), k, center))
    items.sort()

    for _, k, center in items:
        center = (int(center[0]), int(center[1]))
        cv2.drawMarker(output,
                       center,
                       marker_color,
                       markerType=cv2.MARKER_CROSS,
                       markerSize=11,
                       thickness=2)
        text = "{}: ({}, {})".format(k, center[0], center[1])
        place_text(text, center, output)
    return output


def part_1a():

    input_images = ['scene_tl_1']
    output_labels = ['ps2-1-a-1']

    # Define a radii range, you may define a smaller range based on your
    # observations.
    radii_range = range(10, 30, 1)

    for img_in, label in zip(input_images, output_labels):

        tl = cv2.imread("input_images/{}.png".format(img_in))
        coords, state = ps2.traffic_light_detection(tl, radii_range)

        img_out = draw_tl_center(tl, coords, state)
        cv2.imwrite(os.path.join(OUTPUT_DIR,"{}.png".format(label)), img_out)


def part_1b():

    input_images = ['scene_constr_1']

    output_labels = ['ps2-1-b-1']

    sign_fns = [ps2.construction_sign_detection]

    sign_labels = ['construction']

    for img_in, label, fn, name in zip(input_images, output_labels, sign_fns,
                                       sign_labels):

        sign_img = cv2.imread("input_images/{}.png".format(img_in))
        coords = fn(sign_img)

        temp_dict = {name: coords}
        img_out = mark_traffic_signs(sign_img, temp_dict)
        cv2.imwrite(os.path.join(OUTPUT_DIR,"{}.png".format(label)), img_out)


def template_match_test():
    input_images = ['scene_tl_1', 'scene_constr_1', 'waldo_1']
    output_labels = ['ps2-2-a-1', 'ps2-2-b-1', 'ps2-2-c-1']
    templates = [
        'tl_template.png', 'construction_template.png', 'waldo1_template.png'
    ]
    for img_fl, img_template_fl, label in zip(
            input_images,
            templates,
            output_labels,
    ):
        img_in = cv2.imread("input_images/{}.png".format(img_fl))
        img_template = cv2.imread(
            "input_images/{}".format(img_template_fl))

        for method in ("tm_ssd", "tm_nssd", "tm_nccor"):
            """ Convert images to gray scale to save computation """
            top_left = ps2.template_match(img_in, img_template, method)
            bottom_right = None
            """Below is the helper code to print images for the report"""
            im_out = img_in.copy()
            cv2.rectangle(im_out, top_left, bottom_right, 255, 2)
            text = "(({}, {}))".format(top_left[0], top_left[1])
            place_text(text, top_left, im_out)
            cv2.imwrite(os.path.join(OUTPUT_DIR,
                                     "{}-{}.png".format(method, label)),
                        im_out)


def compression_runner():
    img_bgr = cv2.imread(INPUT_DIR + 'dog.jpg', cv2.IMREAD_COLOR)

    # Set the fraction of frequencies to keep for compression
    keep = 0.5

    img_compressed, compressed_frequency_img = ps2.compress_image_fft(
        img_bgr, keep)
    cv2.imwrite(OUTPUT_DIR + 'dog_compressed.jpg', img_compressed)
    cv2.imwrite(OUTPUT_DIR + 'dog_compressed_frequency.jpg',
                compressed_frequency_img)


def low_pass_filter_runner():
    img_bgr = cv2.imread(INPUT_DIR + 'cat.jpg', cv2.IMREAD_COLOR)
    img_bgr = np.ndarray.astype(img_bgr, dtype=np.double)

    # will try 100 here
    radius = 100

    img_low_pass, low_pass_frequency_img_mag = ps2.low_pass_filter(
        img_bgr, radius)

    cv2.imwrite(OUTPUT_DIR + 'cat_lpf.jpg', img_low_pass)
    cv2.imwrite(OUTPUT_DIR + 'cat_lpf_frequency.jpg',
                low_pass_frequency_img_mag)


if __name__ == "__main__":
    # Create Ouput directory
    if not os.path.exists("output_images"):
        os.makedirs("output_images")
    part_1a()
    part_1b()
    template_match_test()
    compression_runner()
    low_pass_filter_runner()
