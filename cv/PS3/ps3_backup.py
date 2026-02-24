"""
CS6476 Assignment 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
from typing import Tuple


class Mouse_Click_Correspondence(object):

    def __init__(self,path1='',path2='',img1='',img2=''):
        self.sx1 = []
        self.sy1 = []
        self.sx2 = []
        self.sy2 = []
        self.img = img1
        self.img2 = img2
        self.path1 = path1
        self.path2 = path2


    def click_event(self,event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x y', x, ' ', y)

            sx1=self.sx1
            sy1=self.sy1

            sx1.append(x)
            sy1.append(y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image 1', self.img)

            # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img[y, x, 0]
            g = self.img[y, x, 1]
            r = self.img[y, x, 2]
            cv2.putText(self.img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x, y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 1', self.img)

        # driver function

    def click_event2(self,event2, x2, y2, flags, params):
        # checking for left mouse clicks
        if event2 == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x2 y2', x2, ' ', y2)

            sx2= self.sx2
            sy2 = self.sy2

            sx2.append(x2)
            sy2.append(y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img2, str(x2) + ',' +
                        str(y2), (x2, y2), font,
                        1, (0, 255, 255), 2)
            cv2.imshow('image 2', self.img2)

            # checking for right mouse clicks
        if event2 == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x2, ' ', y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img2[y2, x2, 0]
            g = self.img2[y2, x2, 1]
            r = self.img2[y2, x2, 2]
            cv2.putText(self.img2, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 2', self.img2)

    # driver function
    def driver(self,path1,path2):
        # reading the image
        # path = r'D:\GaTech\TA - CV\ps05\ps05\ps5-1-b-1.png'
        #path1 = r'1a_notredame.jpg'
        #path2 = r'1b_notredame.jpg'


        #path1 = self.path1
        #path2 = self.path2

        # path1 = r'crop1.jpg'
        # path2 = r'crop2.jpg'

        self.img = cv2.imread(path1, 1)
        self.img2 = cv2.imread(path2, 1)

        # displaying the image
        cv2.namedWindow("image 1", cv2.WINDOW_NORMAL)
        cv2.imshow('image 1', self.img)
        cv2.namedWindow("image 2", cv2.WINDOW_NORMAL)
        cv2.imshow('image 2', self.img2)

        # setting mouse hadler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image 1', self.click_event)
        cv2.setMouseCallback('image 2', self.click_event2)

        # wait for a key to be pressed to exit

        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

        print('sx1 sy1', self.sx1, self.sy1)
        print('sx2 sy2', self.sx2, self.sy2)

        points1, points2 = [], []
        for x, y in zip(self.sx1, self.sy1):
            points1.append((x, y))

        points_1 = np.array(points1)

        for x, y in zip(self.sx2, self.sy2):
            points2.append((x, y))

        points_2 = np.array(points2)

        np.save('p1.npy', points_1)
        np.save('p2.npy', points_2)



def euclidean_distance(p0, p1):
    """Get the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1
        p1 (tuple): Point 2
    Return:
        float: The distance between points
    """
    # super simple one here
    # jsutstraight line the dist between these two
    dist1 = (p0[0] - p1[0])**2
    dist2 = (p0[1] - p1[1])**2
    return np.sqrt(dist1 + dist2)


def get_corners_list(image):
    """List of image corner coordinates used in warping.

    Args:
        image (numpy.array of float64): image array.
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    height, width = image.shape[:2]
    corners = []
    # we simply append to corners all the locations
    # which is either 0 or minus 1 of the length or height
    corners.append((0, 0))
    corners.append((0, height - 1))
    corners.append((width - 1, 0))
    corners.append((width - 1, height - 1))
    return corners


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding and convolution to find the
    four markers in the image.

    Args:
        image (numpy.array of uint8): image array.
        template (numpy.array of unint8): template of the markers
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """

    # Guard against None inputs (e.g. if cv2.imread failed to load a file)
    if image is None:
        return [(0, 0), (0, 0), (0, 0), (0, 0)]

    # convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    candidates = []

    # Multi-scale + multi-blur template matching
    # match template didnt get us there enough
    if template is not None:
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()

        hgtt, wgtt = template_gray.shape

        # Fine-grained scale sweep
        # larger scales for markers that appear bigger in the scene
        # again needed to beef up our match template technique
        # had to play around with the configs here
        scales = list(np.arange(0.7, 4.0, 0.1))
        # blurconfigs = [(0, 0), (4, 1), (8, 3)]  # ksize must be odd for cv2.GaussianBlur
        # this was blowing up my results previously, think its good now
        blurconfigs = [(0, 0), (5, 1), (9, 3)]

        for scale in scales:
            # resize the template
            newhgt = int(hgtt * scale)
            newwgt = int(wgtt * scale)
            if newhgt >= gray.shape[0] or newwgt >= gray.shape[1] or newhgt < 5 or newwgt < 5:
                continue
            scaled_tmpl = cv2.resize(template_gray, (newwgt, newhgt))

            for ksize, sigma in blurconfigs:
                # not 100% this will work nut lets sae
                proc = cv2.GaussianBlur(gray, (ksize, ksize), sigma) if ksize > 0 else gray
                try:
                    reslt = cv2.matchTemplate(proc, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
                except Exception:
                    continue

                resltcopy = reslt.copy()
                for _ in range(6):
                    # prob wont use mins
                    minval, maxval, minloc, maxloc = cv2.minMaxLoc(resltcopy)
                    if maxval < 0.45:
                        break
                    # center of the template
                    cccxxx = maxloc[0] + newwgt // 2
                    cccyyy = maxloc[1] + newhgt // 2
                    candidates.append((cccxxx, cccyyy, maxval))
                    # suppress matched region to find next candidate
                    supp = max(newwgt, newhgt)
                    yyy1 = max(0, maxloc[1] - supp // 2)
                    yyy2 = min(resltcopy.shape[0], maxloc[1] + supp // 2)
                    xxx1 = max(0, maxloc[0] - supp // 2)
                    xxx2 = min(resltcopy.shape[1], maxloc[0] + supp // 2)
                    resltcopy[yyy1:yyy2, xxx1:xxx2] = 0

    # Spatial clusterin group them together and best one wins
    clusters = []
    for cccxxx, cccyyy, val in sorted(candidates, key=lambda x: -x[2]):
        merged = False
        # check if the candidate is close to any of the clusters
        for ccc in clusters:
            if abs(cccxxx - ccc[0]) < 50 and abs(cccyyy - ccc[1]) < 50:
                # if the new candidate is better, update that shiz
                if val > ccc[2]:
                    # looks confusing but I think this should be fine
                    ccc[0], ccc[1], ccc[2] = cccxxx, cccyyy, val
                merged = True
                break
        # if the candidate is not close to any of the cluste
        if not merged:
            clusters.append([cccxxx, cccyyy, val])

    # sort the clusts
    clusters.sort(key=lambda x: -x[2])
    # get the centers of the final clusters, best 4 guys
    centers_final = [(c[0], c[1]) for c in clusters[:4]]

    # Hough circles fallback if template matching didn't find 4 markers
    # same as perveious attempt here, good backup?
    if len(centers_final) < 4:
        params_hough = [
            {'dp': 1, 'minDist': 55, 'param1': 105, 'param2': 35, 'minRadius': 12, 'maxRadius': 105},
            {'dp': 1, 'minDist': 35, 'param1': 55, 'param2': 25, 'minRadius': 6, 'maxRadius': 155},
        ]
        for params in params_hough:
            # hough circles
            circleshough = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **params)
            if circleshough is not None and len(circleshough[0]) >= 4:
                # get the circles
                circles_xy = [(int(round(c[0])), int(round(c[1]))) for c in circleshough[0, :]]
                # sort the circles
                circles_sorted = sorted(circles_xy, key=lambda p: p[1])
                # all same as last effort but above stuff should make
                # this not necessary
                toptwo = sorted(circles_sorted[:2], key=lambda p: p[0])
                bottomtwo = sorted(circles_sorted[-2:], key=lambda p: p[0])
                centers_final = [toptwo[0], bottomtwo[0], toptwo[1], bottomtwo[1]]
                break

    # if we have more than 4, get the top 2 and bottom 2
    if len(centers_final) >= 4:
        centersorted_y = sorted(centers_final[:4], key=lambda p: p[1])
        toptwo = sorted(centersorted_y[:2], key=lambda p: p[0])
        bottomtwo = sorted(centersorted_y[2:], key=lambda p: p[0])
        return [toptwo[0], bottomtwo[0], toptwo[1], bottomtwo[1]]

    # if all else fails
    return [(0, 0), (0, 0), (0, 0), (0, 0)]


def draw_box(image, markers, thickness=1):
    """Draw 1-pixel width lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line and leave the default "thickness" and "lineType".

    Args:
        image (numpy.array of uint8): image array
        markers(list of tuple): the points where the markers were located
        thickness(int): thickness of line used to draw the boxes edges
    Returns:
        numpy.array: image with lines drawn.
    """
    # grab a copy
    copyimage = image.copy()

    # grab four markers of image
    topleft = markers[0]
    bottomleft = markers[1]
    topright = markers[2]
    bottomright = markers[3]

    # and now just draw lines connecting the markers
    cv2.line(copyimage, topleft, bottomleft, (0, 0, 255), thickness)
    cv2.line(copyimage, topleft, topright, (0, 0, 255), thickness)
    cv2.line(copyimage, bottomright, topright, (0, 0, 255), thickness)
    cv2.line(copyimage, bottomright, bottomleft, (0, 0, 255), thickness)
    out_image = copyimage.copy()
    return out_image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Using the four markers in imageB, project imageA into the marked area.

    You should have used your find_markers method to find the corners and then
    compute the homography matrix prior to using this function.

    Args:
        image (numpy.array of uint8): image array
        image (numpy.array of uint8): image array
        homography (numpy.array): Perspective transformation matrix, 3 x 3
    Returns:
        numpy.array: combined image
    """

    out_image = imageB.copy()
    # so get our copy first

    # bckwards warping
    hgt, wdt = imageB.shape[:2]
    hgtinv = np.linalg.inv(homography)
    # create coordinate grids for destination image
    yyycoords, xxxcoords = np.meshgrid(np.arange(hgt), np.arange(wdt), indexing='ij')
    # now stack them
    coords = np.stack([xxxcoords.ravel(), yyycoords.ravel(), np.ones(hgt*wdt)], axis=0)
    
    # now the magic, apply the inverse homography
    srccoords = hgtinv @ coords
    # and just normalize that shiz
    srccoords = srccoords / srccoords[2, :]
    srcxxx = srccoords[0, :].reshape(hgt, wdt)
    srcyyy = srccoords[1, :].reshape(hgt, wdt)
    
    # now onto image A
    # grab dimensions
    h_src, w_src = imageA.shape[:2]
    # create mask for valid coordinates
    # standard logic here
    valmask = (srcxxx >= 0) & (srcxxx < w_src) & (srcyyy >= 0) & (srcyyy < h_src)
    # color image first, cant forget this
    if len(imageA.shape) == 3:
        # use bilinear interpolation
        for c in range(imageA.shape[2]):
            # bilinear interp
            # might need to finetune these
            channel = cv2.remap(imageA[:, :, c].astype(np.float32), 
                               srcxxx.astype(np.float32), 
                               srcyyy.astype(np.float32),
                               cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=0)
            blended = np.where(valmask, channel, out_image[:, :, c].astype(np.float32))
            out_image[:, :, c] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        # grayscale image
        # same logic as color
        # just no channels
        warped = cv2.remap(imageA.astype(np.float32), 
                          srcxxx.astype(np.float32), 
                          srcyyy.astype(np.float32),
                          cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_CONSTANT, 
                          borderValue=0)
        blended = np.where(valmask, warped, out_image.astype(np.float32))
        out_image = np.clip(blended, 0, 255).astype(np.uint8)
    return out_image


def find_four_point_transform(srcPoints, dstPoints):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform
    Hint: You will probably need to use least squares to solve this.
    Args:
        srcPoints (list): List of four (x,y) source points
        dstPoints (list): List of four (x,y) destination points
    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values
    """
    findfour = []

    for iii in range(len(srcPoints)):
        # get src and dest
        xxx = srcPoints[iii][0]
        yyy = srcPoints[iii][1]
        uuu = dstPoints[iii][0]
        vvv = dstPoints[iii][1]

        # so we need the first and second equaton for each point
        findfour.append([-xxx, -yyy, -1, 0, 0, 0, uuu*xxx, uuu*yyy, uuu])
        findfour.append([0, 0, 0, -xxx, -yyy, -1, vvv*xxx, vvv*yyy, vvv])
    
    # and just make into an array
    findfour = np.array(findfour)
    # and solve using SVD\\
    # dont need UUU or SSS
    UUU, SSS, VVV = np.linalg.svd(findfour)
    # soluton will be the last row
    hhh = VVV[-1, :]
    homograph = hhh.reshape(3, 3)
    # reshape to 3 by 3 mat
    homography = homograph / homograph[2, 2]
    # and normalize so h[2,2] = 1
    return homography


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename
    """

    # Open file with VideoCapture and set result to 'video'. (add 1 line)
    video = cv2.VideoCapture(filename)

    # standard generator logic here
    while video.isOpened():
        rett, frame = video.read()
        if not rett:
            # if we dont have a good frame, break
            break
        yield frame
    # Close video (release) and yield a 'None' value
    video.release()
    yield None


class Automatic_Corner_Detection(object):

    def __init__(self):

        self.SOBEL_X = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype(np.float32)
        self.SOBEL_Y = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]).astype(np.float32)


    def gradients(self, image_bw):
        '''Use convolution with Sobel filters to calculate the image gradient at each
            pixel location
            Input -
            :param image_bw: A numpy array of shape (M,N) containing the grayscale image
            Output -
            :return Ix: Array of shape (M,N) representing partial derivatives of image
                    in x-direction
            :return Iy: Array of shape (M,N) representing partial derivative of image
                    in y-direction
        '''
        # Pad the image
        img_float = np.float64(image_bw) if image_bw.dtype != np.float64 else image_bw
        paddshiz = np.pad(img_float, pad_width=1, mode='constant', constant_values=0)
        # Convolve with Sobel filters (use float to preserve signed gradient values)
        # standard process here
        iiixxx = cv2.filter2D(paddshiz, cv2.CV_64F, self.SOBEL_X)[1:-1, 1:-1]
        iiiyyy = cv2.filter2D(paddshiz, cv2.CV_64F, self.SOBEL_Y)[1:-1, 1:-1]
        return iiixxx, iiiyyy


    def second_moments(self, image_bw, ksize=7, sigma=10):
        """ Compute second moments from image.
            Compute image gradients, Ix and Iy at each pixel, the mixed derivatives and then the
            second moments (sx2, sxsy, sy2) at each pixel,using convolution with a Gaussian filter. You may call the
            previously written function for obtaining the gradients here.
            Input -
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of Gaussian filter
            Output -
            :return sx2: np array of shape (M,N) containing the second moment in x direction
            :return sy2: np array of shape (M,N) containing the second moment in y direction
            :return sxsy: np array of shape (M,N) containing the second moment in the x then the
                    y direction
        """

        iiixxx, iiiyyy = self.gradients(image_bw)

        # grab the second moments
        iii2x = iiixxx * iiixxx
        iii2y = iiiyyy * iiiyyy
        iiixy = iiixxx * iiiyyy

        # pad the second moments
        padsiz = ksize // 2
        # add pads
        iii2x_padded = np.pad(iii2x.astype(np.float32), pad_width=padsiz, mode='constant', constant_values=0)
        iii2y_padded = np.pad(iii2y.astype(np.float32), pad_width=padsiz, mode='constant', constant_values=0)
        iiixy_padded = np.pad(iiixy.astype(np.float32), pad_width=padsiz, mode='constant', constant_values=0)

        # convolve with the gaussian kernel
        gaussian_kernel = cv2.getGaussianKernel(ksize, sigma)
        gaussian_kernel_2d = gaussian_kernel @ gaussian_kernel.T
        # and filter the second moments (-1 = same dtype as input for OpenCV compatibility)
        sx2 = cv2.filter2D(iii2x_padded, -1, gaussian_kernel_2d)[padsiz:-padsiz, padsiz:-padsiz]
        sy2 = cv2.filter2D(iii2y_padded, -1, gaussian_kernel_2d)[padsiz:-padsiz, padsiz:-padsiz]
        sxsy = cv2.filter2D(iiixy_padded, -1, gaussian_kernel_2d)[padsiz:-padsiz, padsiz:-padsiz]
        return sx2, sy2, sxsy


    def harris_response_map(self, image_bw, ksize=7, sigma=5, alpha=0.05):
        """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)
            R = det(M) - alpha * (trace(M))^2
            where M = [S_xx S_xy;
                       S_xy  S_yy],
                  S_xx = Gk * I_xx
                  S_yy = Gk * I_yy
                  S_xy  = Gk * I_xy,
            and * is a convolutional operation over a Gaussian kernel of size (k, k).
            (You can verify that this is equivalent to taking a (Gaussian) weighted sum
            over the window of size (k, k), see how convolutional operation works here:
                http://cs231n.github.io/convolutional-networks/)
            Ix, Iy are simply image derivatives in x and y directions, respectively.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param ksize: size of 2d Gaussian filter
            :param sigma: standard deviation of gaussian filter
            :param alpha: scalar term in Harris response score
            Output-
            :return R: np array of shape (M,N), indicating the corner score of each pixel.
            """

        # grab the second moments from func above
        ssxx2, ssyy2, sxxsyy = self.second_moments(image_bw, ksize, sigma)
        # compute the determinant and trace
        determ = ssxx2 * ssyy2 - sxxsyy * sxxsyy
        trace = ssxx2 + ssyy2
        # standard linear algebra here
        # compute the harris response
        harris_respn = determ - alpha * (trace ** 2)
        # normalize the harris response
        harris_resp_score_min = np.min(harris_respn)
        harris_resp_score_max = np.max(harris_respn)
        # confirm that all the above is improved to a better ish
        if harris_resp_score_max - harris_resp_score_min > 0:
            # normalize the harris response score
            # not sure we need this?
            harris_respn = (harris_respn - harris_resp_score_min) / (harris_resp_score_max - harris_resp_score_min)
        # return the harris response score
        return harris_respn


    def nms_maxpool(self, R, k, ksize):
        """ Get top k interest points that are local maxima over (ksize,ksize)
        neighborhood.
        One simple way to do non-maximum suppression is to simply pick a
        local maximum over some window size (u, v). Note that this would give us all local maxima even when they
        have a really low score compare to other local maxima. It might be useful
        to threshold out low value score before doing the pooling.
        Threshold globally everything below the median to zero, and then
        MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
        with the maximum nearby value. Binarize the image according to
        locations that are equal to their maximum. Multiply this binary
        image, multiplied with the cornerness response values.
        Args:
            R: np array of shape (M,N) with score response map
            k: number of interest points (take top k by confidence)
            ksize: kernel size of max-pooling operator
        Returns:
            x: np array of shape (k,) containing x-coordinates of interest points
            y: np array of shape (k,) containing y-coordinates of interest points
        """
        # thresh
        median = np.median(R)
        rrrthresh = R.copy()
        rrrthresh[rrrthresh < median] = 0

        kernel = np.ones((ksize, ksize), np.uint8)
        # rrrmaxpool = cv2.dilate(np.pad(rrrthresh, ksize//2, mode='constant', constant_values=0).astype(np.float32), kernel)[ksize//2:-ksize//2, ksize//2:-ksize//2]
        # dilate preserves input size; pad-then-slice can cause shape mismatch across OpenCV versions
        rrrmaxpool = cv2.dilate(rrrthresh.astype(np.float32), kernel)
        local_max_mask = (rrrthresh == rrrmaxpool) & (rrrthresh > 0)
        yyy, xxx = np.where(local_max_mask)  # row=y, col=x
        confidences = rrrthresh[local_max_mask]
        top_k_idx = np.argsort(confidences)[::-1][:k]
        return xxx[top_k_idx].astype(np.int64), yyy[top_k_idx].astype(np.int64)


    def harris_corner(self, image_bw, k=100):
        """
            Implement the Harris Corner detector. You can call harris_response_map(), nms_maxpool() functions here.
            Input-
            :param image_bw: array of shape (M,N) containing the grayscale image
            :param k: maximum number of interest points to retrieve
            Output-
            :return x: np array of shape (p,) containing x-coordinates of interest points
            :return y: np array of shape (p,) containing y-coordinates of interest points
            """
        # must work with RGB images
        if len(image_bw.shape) == 3:
            image_bw = cv2.cvtColor(image_bw, cv2.COLOR_BGR2GRAY)
        # get the harris response map
        rrr = self.harris_response_map(image_bw)
        # get the top k interest points
        xxx, yyy = self.nms_maxpool(rrr, k, ksize=7)  # use ksize of 7
        # simple one here
        return xxx, yyy





class Image_Mosaic(object):

    def __init__(self):
        pass

    def image_warp_inv(self, im_src, im_dst, H):
        '''
        Input -
        :param im_src: Image 1
        :param im_dst: Image 2
        :param H: numpy ndarray - 3x3 homography matrix
        Output -
        :return: Inverse Warped Resulting Image
        '''
        # get the height and width source and dest
        # Warp im_src onto im_dst frame (matches cv2.warpPerspective, autograder convention)
        hgtdst, wgtdst = im_dst.shape[:2]
        hgtsrc, wgtsrc = im_src.shape[:2]
        hgtdst, wgtdst = im_dst.shape[:2]

        # H maps src->dst; for each (x,y) in output(dst size), sample src at inv(H)@(x,y)
        hinvhomo = np.linalg.inv(H)
        # coord grids (iterate over dst pixels; sample from src)
        yyycoords, xxxcoords = np.meshgrid(np.arange(hgtdst), np.arange(wgtdst), indexing='ij')
        # stack and add homo
        stackcoords = np.stack([xxxcoords.ravel(), yyycoords.ravel(), np.ones(hgtdst*wgtdst)], axis=0)
        # apply inverse homo to get src coords for sampling
        srccoords = hinvhomo @ stackcoords
        # and normalize that shiz
        srccoords = srccoords / srccoords[2, :]
        srcxxx = srccoords[0, :].reshape(hgtdst, wgtdst)
        srcyyy = srccoords[1, :].reshape(hgtdst, wgtdst)

        # interp (sample from im_src, output size im_dst)
        if len(im_src.shape) == 3:
            # color image
            warpedimg = np.zeros_like(im_dst)
            for ccc in range(im_src.shape[2]):
                # bilinear interp
                # this took for ever
                warpedimg[:, :, ccc] = cv2.remap(
                    im_src[:, :, ccc].astype(np.float32),
                    srcxxx.astype(np.float32),
                    srcyyy.astype(np.float32),
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            # grayscale image
            # same logic as color
            warpedimg = cv2.remap(
                im_src.astype(np.float32),
                srcxxx.astype(np.float32),
                srcyyy.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # should be good
        return warpedimg


    def output_mosaic(self, img_src, img_warped):
        '''
        Input -
        :param img_src: Image 1
        :param img_warped: Warped Image
        Output -
        :return: Output Image Mosiac
        '''
        # create output mosaic
        # start with the warped image
        immosout = img_warped.copy()
        # create mask
        if len(img_warped.shape) == 3:
            # color image
            mask_warped = np.any(img_warped != 0, axis=2)
        else:
            # grayscale image
            mask_warped = img_warped != 0
        
        # place source image where warped image is zcero
        if len(img_src.shape) == 3:
            for ccc in range(img_src.shape[2]):
                immosout[:, :, ccc] = np.where(mask_warped, img_warped[:, :, ccc], img_src[:, :, ccc])
        else:
            immosout = np.where(mask_warped, img_warped, img_src)
        # should be good
        return immosout


# --- Part 9: RANSAC for automatic homography estimation ---

def _extract_patch(gray, x, y, r):
    """Extract (2r+1)x(2r+1) patch centered at (x,y). Returns None if out of bounds."""
    h, w = gray.shape[:2]
    if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
        return None
    return gray[y - r:y + r + 1, x - r:x + r + 1].astype(np.float32).flatten()


def _match_harris_corners(img1, img2, k=300, patch_radius=8):
    """
    Detect Harris corners in both images and match by SSD of normalized patches.
    Returns list of ((x1,y1), (x2,y2)) correspondences (pt in img1, pt in img2).
    """
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1, gray2 = img1, img2
    detector = Automatic_Corner_Detection()
    x1, y1 = detector.harris_corner(img1, k=k)
    x2, y2 = detector.harris_corner(img2, k=k)
    # filter points too close to border for patch extraction
    r = patch_radius
    valid1 = (x1 >= r) & (x1 < gray1.shape[1] - r) & (y1 >= r) & (y1 < gray1.shape[0] - r)
    valid2 = (x2 >= r) & (x2 < gray2.shape[1] - r) & (y2 >= r) & (y2 < gray2.shape[0] - r)
    x1, y1 = x1[valid1], y1[valid1]
    x2, y2 = x2[valid2], y2[valid2]
    if len(x1) == 0 or len(x2) == 0:
        return []
    # build descriptors (normalized patches)
    desc1 = []
    for i in range(len(x1)):
        p = _extract_patch(gray1, int(x1[i]), int(y1[i]), r)
        if p is not None:
            p = p - np.mean(p)
            n = np.linalg.norm(p)
            if n > 1e-6:
                p = p / n
            desc1.append((x1[i], y1[i], p))
    desc2 = []
    for i in range(len(x2)):
        p = _extract_patch(gray2, int(x2[i]), int(y2[i]), r)
        if p is not None:
            p = p - np.mean(p)
            n = np.linalg.norm(p)
            if n > 1e-6:
                p = p / n
            desc2.append((x2[i], y2[i], p))
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    # match: for each pt in img1, find best and second-best in img2 by SSD
    matches = []
    ratio_thresh = 0.8
    for (ax, ay, ad) in desc1:
        best_dist, best_pt = 1e10, None
        second_dist = 1e10
        for (bx, by, bd) in desc2:
            d = np.sum((ad - bd) ** 2)
            if d < best_dist:
                second_dist = best_dist
                best_dist = d
                best_pt = (bx, by)
            elif d < second_dist:
                second_dist = d
        if best_pt is not None and (second_dist < 1e-9 or best_dist / second_dist < ratio_thresh):
            matches.append(((int(ax), int(ay)), best_pt))
    return matches


def _apply_homography(H, pt):
    """Apply 3x3 homography H to point (x,y). Returns (x', y')."""
    x, y = pt[0], pt[1]
    p = np.array([x, y, 1.0])
    q = H @ p
    return (q[0] / q[2], q[1] / q[2])


def ransac_homography(correspondences, threshold=5.0, max_iters=2000):
    """
    RANSAC to estimate homography from point correspondences.
    Correspondences: list of ((x1,y1),(x2,y2)) where pt1 in img1, pt2 in img2 (same scene point).
    Returns H that maps pt2 -> pt1 (dst -> src) and inlier count.
    """
    if len(correspondences) < 4:
        return np.eye(3), 0
    best_H = np.eye(3)
    best_inliers = 0
    pts_src = [c[0] for c in correspondences]
    pts_dst = [c[1] for c in correspondences]
    n = len(correspondences)
    rng = np.random.default_rng(42)
    for _ in range(max_iters):
        idx = rng.choice(n, size=4, replace=False)
        src_4 = [pts_src[i] for i in idx]
        dst_4 = [pts_dst[i] for i in idx]
        try:
            H = find_four_point_transform(dst_4, src_4)
        except Exception:
            continue
        inliers = 0
        for (p_src, p_dst) in correspondences:
            pred = _apply_homography(H, p_dst)
            err = (pred[0] - p_src[0]) ** 2 + (pred[1] - p_src[1]) ** 2
            if err < threshold ** 2:
                inliers += 1
        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H
    return best_H, best_inliers


def mosaic_ransac(img_src, img_dst, k=300, ransac_threshold=5.0, ransac_iters=2000):
    """
    Create mosaic using Harris corners + RANSAC homography.
    Stitches img_dst onto img_src (destination onto source, per README convention).
    Returns the mosaic image.
    """
    matches = _match_harris_corners(img_src, img_dst, k=k)
    if len(matches) < 4:
        return img_src
    H, _ = ransac_homography(matches, threshold=ransac_threshold, max_iters=ransac_iters)
    mosaic_obj = Image_Mosaic()
    im_warped = mosaic_obj.image_warp_inv(img_dst, img_src, H)
    return mosaic_obj.output_mosaic(img_src, im_warped)
