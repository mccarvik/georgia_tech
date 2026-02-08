import cv2
import numpy as np
from matplotlib import pyplot as plt


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.
    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.
    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.
    It is recommended you use Hough tools to find these circles in
    the image.
    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.
    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.
    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    # this gets us hte gray scalle copy
    if len(img_in.shape) == 3:
        gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_in.copy()
    # we want a clean copy 
    
    # add some blure to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # so now we use the Hough Transform to finde the circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=radii_range[0], maxRadius=radii_range[1])
    
    if circles is None:
        # didnt find any
        return (0, 0), 'nada'

    circs = np.uint16(np.around(circles)) 
    circs_list = []
    # setting up the list of circles we find

    for circle in circs[0, :]:
        x, y, r = circle
        circs_list.append((x, y, r))
    # Sort circles by y-coordinate (top to bottom)
    circs_list.sort(key=lambda c: c[1])

    # we only need 3 circles
    if len(circs_list) < 3:
        if len(circs_list) > 0:
            # find brightest one
            max_brite = 0
            state = 'nada'
            y_cent = 0
            # grab 3 vals for the circs
            for xxx, yyy, rrr in circs_list:
                mask = np.zeros(gray.shape[:2])
                # apply the mask
                cv2.circle(mask, (xxx, yyy), rrr, 255, -1)
                # grab that mean
                mean_val = cv2.mean(gray, mask=mask)[0]

                if mean_val > max_brite:
                    # standard shiz here
                    max_brite = mean_val
                    y_cent = yyy
                    
            # use mid y for centter
            # im really not sure this is gonna work, will come back to this
            center_x = circs_list[0][0]
            # i guess we just need middle here for hte x, will be the same for all cuz its a traffic light
            y_cent = circs_list[len(circs_list)//2][1] if len(circs_list) >= 2 else circs_list[0][1]
            return (int(center_x), int(y_cent)), state
    
    # but if we have at least 3 circles, traffic light detect
    if len(circs_list) >= 3:
        # red (top), yellow (middle), green (bottom)
        red_circ = circs_list[0]
        yellow_circ = circs_list[1]
        green_circ = circs_list[2]
        
        # Center of traffic light is at the yellow circle... probably
        center_x, center_y = yellow_circ[0], yellow_circ[1]

        max_intens = 0
        state = 'nada'

        for idx, (xxx, yyy, rrr) in enumerate([red_circ, yellow_circ, green_circ]):
            # go thru each
            # make a circular mask
            mask = np.zeros(gray.shape[:2])
            # apply the mask LIKE UP TOP
            cv2.circle(mask, (xxx, yyy), max(1, rrr-2), 255, -1)
            # again grab that mean
            mean_val = cv2.mean(gray, mask=mask)[0]

            if mean_val > max_intens:
                max_intens = mean_val
                if idx == 0:
                    state = 'red'
                elif idx == 1:
                    state = 'yellow'
                else:
                    state = 'green'
            # in theory, max intensity is hte one thats lit

        return (int(center_x), int(center_y)), state
    return (0, 0), 'nada'


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.
    Args:
        img_in (numpy.array): image containing a traffic light.
    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    # gray scale that shiz
    if len(img_in.shape) == 3:
        gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_in.copy()
    # get a copy either way of the param

    # apply edge detection
    # lets try the cv2 edge detect
    # might have to finetune these more
    edges = cv2.Canny(gray, 30, 125, apertureSize=4)
    # Hough Line Transform
    # really not sure on half these inputs but well see
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=20, maxLineGap=5)

    # didnt find any lines
    if not lines:
        return (0, 0)
    
    # set to empty
    all_points = []
    # grab 45 degree lines
    lines_45_angle = {'45': [], '-45': []}
    
    # these 45s are crucial, I think this should work but it may get false positives of other signs
    for line in lines:
        xx1, yy1, xx2, yy2 = line[0]
        # i think this should get me the angle
        angle = np.arctan2(yy2 - yy1, xx2 - xx1)
        # some range for the 45 line
        if 30 < angle < 60 or -150 < angle < -120:
            lines_45_angle['45'].append(line[0])
        elif -60 < angle < -30 or 120 < angle < 150:
            lines_45_angle['-45'].append(line[0])

    # If we have both 45 lines, this is prob construction right?
    if len(lines_45_angle['45']) > 0 and len(lines_45_angle['-45']) > 0:
        # Calculat center
        all_points = np.array(all_points)
        # filter points
        angle_points = []
        for line in lines_45_angle['45'] + lines_45_angle['-45']:
            xx1, yy1, xx2, yy2 = line[0]
            angle_points.extend([(xx1, yy1), (xx2, yy2)])
            # grab all the points
        
        # this should be the center of the diamond
        if len(angle_points) > 0:
            angle_points = np.array(angle_points)
            centroid_x = int(np.mean(angle_points[:, 0]))
            centroid_y = int(np.mean(angle_points[:, 1]))
            return (int(centroid_x), int(centroid_y))
        # otherwise just return 0
        return (0, 0)


def template_match(img_orig, img_template, method):
    """Returns the location corresponding to match between original image and provided template.
    Args:
        img_orig (np.array) : numpy array representing 2-D image on which we need to find the template
        img_template: numpy array representing template image which needs to be matched within the original image
        method: corresponds to one of the four metrics used to measure similarity between template and image window
    Returns:
        Co-ordinates of the topmost and leftmost pixel in the result matrix with maximum match
    """
    """Each method is calls for a different metric to determine
       the degree to which the template matches the original image
       We are required to implement each technique using the
       sliding window approach.
       Suggestion : For loops in python are notoriously slow
       Can we find a vectorized solution to make it faster?
    """
    # set up the result matrix
    # set up the result matrix
    first_entry = img_orig.shape[0] - img_template.shape[0] + 1
    second_entry = img_orig.shape[1] - img_template.shape[1] + 1
    result = np.zeros((first_entry, second_entry), float)
    top_left = []
    """Once you have populated the result matrix with the similarity metric corresponding to each overlap, return the topmost and leftmost pixel of
    the matched window from the result matrix. You may look at Open CV and numpy post processing functions to extract location of maximum match"""
   
   # set up the template and result
    h_templ, w_templ = img_template.shape[:2]
    h_res, w_res = result.shape

    # Sum of squared differences
    if method == "tm_ssd":
        """Your code goes here"""
        # damn right it does
        for i in range(h_res):
            for j in range(w_res):
                # get the wind
                window = img_orig[i:i+h_templ, j:j+w_templ]
                # get the diff
                diff = window.astype(float) - img_template.astype(float)
                result[i, j] = np.sum(diff ** 2)
        # For SSD, min is best
        # grab the top left corner of it
        min_idx = np.argmin(result) 
        top_left = np.unravel_index(min_idx, result.shape)
        top_left = (top_left[1], top_left[0])  # Convert to (x, y) format

    # Normalized sum of squared differences
    elif method == "tm_nssd":
        """Your code goes here"""
        # again darn right
        for i in range(h_res):
            for j in range(w_res):
                window = img_orig[i:i+h_templ, j:j+w_templ]
                template = img_template.astype(float)
                # same window as above, well prob use this more below
                # easier to break this up
                numer = np.sum((window - template) ** 2)
                denom = np.sqrt(np.sum(window ** 2) * np.sum(template ** 2))
                # quick error check
                if denom == 0:
                    result[i, j] = np.inf
                else:
                    result[i, j] = numer / denom

        # For NSSD, min is best
        # same as above to grab the index
        min_idx = np.argmin(result)
        top_left = np.unravel_index(min_idx, result.shape)
        top_left = (top_left[1], top_left[0])  # Convert to (x, y) format

    # Normalized Cross Correlation
    elif method == "tm_nccor":
        """Your code goes here"""
        # yup
        # grab these for later
        mean_template = np.mean(img_template)
        std_template = np.std(img_template)

        for i in range(h_res):
            for j in range(w_res):
                window = img_orig[i:i+h_templ, j:j+w_templ]
                template = img_template.astype(float)
                # same as above, CnP
                mean_window = np.mean(window)
                std_window = np.std(window)
                # quick 0 check
                if std_window == 0 or std_template == 0:
                    result[i, j] = 0
                else:
                    # save result
                    norm_window = (window - mean_window) / std_window
                    norm_template = (template - mean_template) / std_template
                    result[i, j] = np.mean(norm_window * norm_template)

        # again third time here
        max_idx = np.argmax(result)
        top_left = np.unravel_index(max_idx, result.shape)
        top_left = (top_left[1], top_left[0])  # Convert to (x, y) format
    else:
        """Your code goes here"""
        # Invalid technique
        raise ValueError("Invalid method. Use one of these guys bro: 'tm_ssd', 'tm_nssd', or 'tm_nccor'")
    return top_left


'''Below is the helper code to print images for the report'''
#     cv2.rectangle(img_orig,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(result,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img_orig,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(method)
#     plt.show()


def dft(x):
    """Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing Fourier Transformed Signal

    """
    x = np.asarray(x, dtype=np.complex128)
    nnn = x.shape[0]
    # just to get started
    # i think this is right, had to look this up
    www = np.exp(-2j * np.pi / nnn)
    # so we need j,k = w ** (j * k)
    jjj = np.arange(nnn).reshape(-1, 1)
    kkk = np.arange(nnn).reshape(1, -1)
    mmm = np.power(www, jjj * kkk)

    # so now we need to do the discrete fourier transform
    yyy = np.dot(mmm, x)
    return yyy


def idft(x):
    """Inverse Discrete Fourier Transform for 1D signal
    Args:
        x (np.array): 1-dimensional numpy array of shape (n,) representing Fourier-Transformed signal
    Returns:
        y (np.array): 1-dimensional numpy array of shape (n,) representing signal

    """
    # So this is gonna look very similar to above
    x = np.asarray(x, dtype=np.complex128)
    nnn = x.shape[0]
    # just to get started
    # i think this is right, had to look this up
    # typo here on the neg exponent
    www = np.exp(2j * np.pi / nnn)
    # so we need j,k = w ** (j * k)
    jjj = np.arange(nnn).reshape(-1, 1)
    kkk = np.arange(nnn).reshape(1, -1)
    mmm = np.power(www, jjj * kkk)

    # so now we need to do the discrete fourier transform
    # yyy = np.dot(mmm, x)
    # this is the change
    # this is the inverse dft
    yyy = (1.0 / nnn) * np.dot(mmm, x)
    return yyy


def dft2(img):
    """Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image

    """
    # take each row and dft it
    rowdfter = np.array([dft(row) for row in img])
    # now take each column and dft it
    coldfter = np.array([dft(rowdfter[:, i]) for i in range(rowdfter.shape[1])]).T
    return coldfter


def idft2(img):
    """Inverse Discrete Fourier Transform for 2D signal
    Args:
        img (np.array): 2-dimensional numpy array of shape (n,m) representing Fourier-Transformed image
    Returns:
        y (np.array): 2-dimensional numpy array of shape (n,m) representing image

    """
    # gonna be like same as before except inverse now
    # take each row and idft it
    # no need to fix exponent ere
    rowidfter = np.array([idft(row) for row in img])
    # now take each column and idft it
    colidfter = np.array([idft(rowidfter[:, i]) for i in range(rowidfter.shape[1])]).T
    return colidfter


def compress_image_fft(img_bgr, threshold_percentage):
    """Return compressed image by converting to fourier domain, thresholding based on threshold percentage, and converting back to fourier domain
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        threshold_percentage (float): between 0 and 1 representing what percentage of Fourier image to keep
    Returns:
        img_compressed (np.array): numpy array of shape (n,m,3) representing compressed image. (Make sure the data type of the np array is float64)
        compressed_frequency_img (np.array): numpy array of shape (n,m,3) representing the compressed image in the frequency domain

    """
    # start here for later to be filled
    img_compress = np.zeros_like(img_bgr, dtype=np.float64)
    # need to fix complex np bug here
    compress_freq_img = np.zeros_like(img_bgr, dtype=np.complex128)

    # process each channel sep
    for channel in range(3):
        # only 3 chans
        # use our fft2
        freq_img = dft2(img_bgr[:, :, channel])
        # get mag
        mag = np.abs(freq_img)
        # flatten and sort
        flat_mag = mag.flatten()
        sort_mag = np.sort(flat_mag)
        thresh_mag = np.percentile(flat_mag, threshold_percentage * 100)
        # make mask
        mask = mag >= thresh_mag
        # apply mask to freq img
        compressed_freq = freq_img * mask
        # convert back
        img_channel = idft2(compressed_freq)
        # USE OUR func
        img_compress[:, :, channel] = np.real(img_channel)
        compress_freq_img[:, :, channel] = compressed_freq
    return img_compress, compress_freq_img


def low_pass_filter(img_bgr, r):
    """Return low pass filtered image by keeping a circle of radius r centered on the frequency domain image
    Args:
        img_bgr (np.array): numpy array of shape (n,m,3) representing bgr image
        r (float): radius of low pass circle
    Returns:
        img_low_pass (np.array): numpy array of shape (n,m,3) representing low pass filtered image. (Make sure the data type of the np array is float64)
        low_pass_frequency_img (np.array): numpy array of shape (n,m,3) representing the low pass filtered image in the frequency domain

    """
    # alst one, finally
    # same setup as above, gonna do a bunch of CnP
    img_low_pass = np.zeros_like(img_bgr, dtype=np.float64)
    low_pass_freq_img = np.zeros_like(img_bgr, dtype=np.complex128)

    orig_rows, orig_cols = img_bgr.shape[:2]
    center_row, center_col = orig_rows // 2, orig_cols // 2
    # make mask
    mask = np.zeros((orig_rows, orig_cols), dtype=np.complex128)
    mask[center_row-r:center_row+r, center_col-r:center_col+r] = 1

    # process each channel sep
    for channel in range(3):
        # only 3 chans
        freq_img = dft2(img_bgr[:, :, channel])
        # set up rows and cols
        rows, cols = freq_img.shape
        rowshift, colshift = rows // 2, cols // 2

        # shift for each to centet
        freq_img_shifted = np.empty_like(freq_img)
        freq_img_shifted[:rowshift, :colshift] = freq_img[rowshift:, colshift:]
        freq_img_shifted[:rowshift, colshift:] = freq_img[rowshift:, :colshift]
        freq_img_shifted[rowshift:, :colshift] = freq_img[:rowshift, colshift:]
        freq_img_shifted[rowshift:, colshift:] = freq_img[:rowshift, :colshift]
        # grab cirular mask
        filt_freq = freq_img_shifted * mask
        # impl inverse shift
        filt_freq_unshifted = np.empty_like(filt_freq)
        filt_freq_unshifted[:rowshift, :colshift] = filt_freq[rowshift:, colshift:]
        filt_freq_unshifted[:rowshift, colshift:] = filt_freq[rowshift:, :colshift]
        filt_freq_unshifted[rowshift:, :colshift] = filt_freq[:rowshift, colshift:]
        filt_freq_unshifted[rowshift:, colshift:] = filt_freq[:rowshift, :colshift]

        # convert back
        img_channel = idft2(filt_freq_unshifted)
        img_low_pass[:, :, channel] = np.real(img_channel)
        low_pass_freq_img[:, :, channel] = filt_freq

        # zero idea if this is gonna work
    return img_low_pass, low_pass_freq_img