"""Problem Set 4: Motion Detection"""

import cv2
import numpy as np


# Burt-Adelson 5-tap generating kernel (a=0.4): ŵ = [0.05, 0.25, 0.40, 0.25, 0.05]
# might change this but keeping it here for now
REDUCE_KERNEL_1D = np.array([0.05, 0.25, 0.40, 0.25, 0.05], dtype=np.float64)
# might need this later
EXPAND_KERNEL_1D = 2.0 * REDUCE_KERNEL_1D

# Utility function
def read_video(video_file, show=False):
    """Reads a video file and outputs a list of consecuative frames
  Args:
      image (string): Video file path
      show (bool):    Visualize the input video. WARNING doesn't work in
                      notebooks
  Returns:
      list(numpy.ndarray): list of frames
  """
    frames = []
    cap = cv2.VideoCapture(video_file)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        # Opens a new window and displays the input
        if show:
            cv2.imshow("input", frame)
            # Frames are read by intervals of 1 millisecond. The
            # programs breaks out of the while loop when the
            # user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # The following frees up resources and
    # closes all windows
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return frames


def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    # set params
    ksize = 3
    scale = 1.3
    ret = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale)
    return ret


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    # set params
    # same as above
    ksize = 3
    scale = 1.3
    ret = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale)
    return ret


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    # get gradients
    i_xxx = gradient_x(img_a)
    i_yyy = gradient_y(img_a)
    # get time derivative
    i_ttt = img_b.astype(np.float64) - img_a.astype(np.float64)

    # grab all the second derivaties
    i_2xxx = i_xxx * i_xxx
    i_2yyy = i_yyy * i_yyy
    i_xxyy = i_xxx * i_yyy
    i_xxtt = i_xxx * i_ttt
    i_yytt = i_yyy * i_ttt

    # get our kernel
    if k_type == 'uniform':
        kernel = np.ones((k_size, k_size))
        kernel = kernel / (k_size * k_size)
    else:
        kernel = cv2.getGaussianKernel(k_size, sigma)
        kernel = kernel * kernel.T

    # convolve all of our stuff with our kernel
    sum_i_2xxx = cv2.filter2D(i_2xxx, cv2.CV_64F, kernel)
    sum_i_2yyy = cv2.filter2D(i_2yyy, cv2.CV_64F, kernel)
    sum_i_xxyy = cv2.filter2D(i_xxyy, cv2.CV_64F, kernel)
    sum_i_xxtt = cv2.filter2D(i_xxtt, cv2.CV_64F, kernel)
    sum_i_yytt = cv2.filter2D(i_yytt, cv2.CV_64F, kernel)

    # calculate our determinants
    det = sum_i_2xxx * sum_i_2yyy - sum_i_xxyy * sum_i_xxyy
    # handle division by zero
    det = np.where(np.abs(det) < 1e-9, np.nan, det)
    uuu = (-sum_i_2yyy * sum_i_xxtt + sum_i_xxyy * sum_i_yytt)
    uuu = uuu / det
    vvv = (-sum_i_xxyy * sum_i_xxtt + sum_i_2xxx * sum_i_yytt)
    vvv = vvv / det
    # handle division by zero
    # this saved a bunch of errors
    uuu = np.nan_to_num(uuu, nan=0.0, posinf=0.0, neginf=0.0)
    vvv = np.nan_to_num(vvv, nan=0.0, posinf=0.0, neginf=0.0)
    return uuu, vvv


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    # convert to float64
    img = np.asarray(image, dtype=np.float64)
    # might have to edit this guy, but well see how it goes
    kernel = REDUCE_KERNEL_1D.reshape(1, -1)

    # alright now we do our convolution
    # the bordertype might cahnge later, but seemed to work for now
    tmp_convolve_1 = cv2.filter2D(img, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT101)
    tmp_convolve_2 = cv2.filter2D(tmp_convolve_1, cv2.CV_64F, kernel.T, borderType=cv2.BORDER_REFLECT101)
    # return the image
    # subsample the image
    ret_stuff = tmp_convolve_2[::2, ::2].copy()
    return ret_stuff


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in 
    [0.0, 1.0].

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    # grab the data off of the image
    pyram_dat = [np.asarray(image, dtype=np.float64).copy()]
    # loop tru each level
    for lev in range(levels - 1):
        # reduce the image
        add_lev = reduce_image(pyram_dat[-1])
        pyram_dat.append(add_lev)
    return pyram_dat


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    # use our normalize function to start
    normal_dat = [normalize_and_scale(np.asarray(ind_img, dtype=np.float64)) for ind_img in img_list]
    # get the height of the largest image
    hhh = max(ind_img.shape[0] for ind_img in normal_dat)
    # get the width of the largest image
    www = max(ind_img.shape[1] for ind_img in normal_dat)
    # now weved got the size of the largest img
    out_img = np.zeros((hhh, www), dtype=np.uint8)

    xxx = 0
    for ind_img in normal_dat:
        # get h and w of cur image
        new_hgt, new_wdt = ind_img.shape
        # copy the image to the output
        out_img[:new_hgt, xxx:xxx + new_wdt] = ind_img
        # and now we update the x coordinate
        xxx += new_wdt

    return out_img


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """

    raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """

    raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    # grab image shape
    mmm, nnn = image.shape
    # map the x an y coordss
    xxx, yyy = np.meshgrid(np.arange(nnn, dtype=np.float32), np.arange(mmm, dtype=np.float32))
    mapxxx = (xxx + U.astype(np.float32))
    mapyyy = (yyy + V.astype(np.float32))

    # and now we remap
    # params given to us
    warped = cv2.remap(image, mapxxx, mapyyy, interpolation, borderMode=border_mode)
    return warped


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    # grab the gaussian pyramids
    # using our builtin function
    gauss_pyrdat_a = gaussian_pyramid(img_a, levels)
    gauss_pyrdat_b = gaussian_pyramid(img_b, levels)

    # so this is gonna grab the coarsest level
    uuu = np.zeros_like(gauss_pyrdat_a[-1])
    vvv = np.zeros_like(gauss_pyrdat_a[-1])

    # loop thru each level
    # start from coarests and go backwards
    for level in range(levels-1, -1, -1):
        # grab each
        level_dat_a = gauss_pyrdat_a[level]
        level_dat_b = gauss_pyrdat_b[level]
        # basically we are warping a toward b using leveldata
        if level < levels-1:
            # now scale up the image if we need to
            uuu = 2.0 * expand_image(uuu)
            vvv = 2.0 * expand_image(vvv)
            # crop to current level size if needed
            hhh, www = level_dat_a.shape
            if uuu.shape[0] != hhh or uuu.shape[1] != www:
                uuu = uuu[:hhh, :www]
            if vvv.shape[0] != hhh or vvv.shape[1] != www:
                vvv = vvv[:hhh, :www]
            # warp b_lev by (u,v) to align with a_lev


def classify_video(images):
    """Classifies a set of frames as either
        - int(1) == "Running"
        - int(2) == "Walking"
        - int(3) == "Clapping"
    Args:
        images list(numpy.array): greyscale floating-point frames of a video
    Returns:
        int:  Class of video
    """

    raise NotImplementedError
