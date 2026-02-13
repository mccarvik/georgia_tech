"""
CS6476 Assignment 3 imports. Only Numpy and cv2 are allowed.
"""
from gettext import find
import cv2
import numpy as np

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

    # so to start, convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # weve got a copy of the image either way

    centers = []
    # if we have a tampleate, use it
    if template is not None:
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
        
        reslt = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        hgt, wdt = template_gray.shape
        threshold = 0.5
        temp_centers = []
        reslt_copy = reslt.copy()

        for _ in range(18):
            # find up to 18 candidates, this seems like a reasonable amount
            # not sure we need the min vals
            minval, maxval, minloc, maxloc = cv2.minMaxLoc(reslt_copy)
            if maxval < threshold:
                # if we dont have a good match, break
                break
            # now lets grab the center of each match
            # stadard math here
            centx = maxloc[0] + wdt // 2
            centy = maxloc[1] + hgt // 2
            # append it (include maxval for sorting by confidence)
            temp_centers.append((centx, centy, maxval))

            # Suppress the region
            suppress_ratio = 0.8
            suppress_size = int(max(wdt, hgt) * suppress_ratio)
            yyy1 = max(0, maxloc[1] - suppress_size // 2)
            yyy2 = min(reslt_copy.shape[0], maxloc[1] + suppress_size // 2)
            xxx1 = max(0, maxloc[0] - suppress_size // 2)
            xxx2 = min(reslt_copy.shape[1], maxloc[0] + suppress_size // 2)
            reslt_copy[yyy1:yyy2, xxx1:xxx2] = 0
            # this should get the suppressed copy

        centers_final = []
        if len(temp_centers) >= 4:
            # Sort by confidence and take top 4
            # we dont need more than 4 centers
            temp_centers.sort(key=lambda x: x[2], reverse=True)
            centers_final = [(c[0], c[1]) for c in temp_centers[:4]]

        # use houghcircles if template didnt do it
        if len(centers_final) < 4:
            # dont hae 4 centers
            params_hough = [
                {'dp': 1, 'minDist': 55, 'param1': 105, 'param2': 35, 'minRadius': 12, 'maxRadius': 105},
                {'dp': 1, 'minDist': 35, 'param1': 55, 'param2': 25, 'minRadius': 6, 'maxRadius': 155},
            ]
            circleshough = None
            for params in params_hough:
                circleshough = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **params)
                if circleshough is not None and len(circleshough[0]) >= 4:
                    break
                    # if we got em, we good
            if circleshough is not None and len(circleshough[0]) >= 4:
                # again, choose top 4
                circleshough_sorted = sorted(circleshough[0, :], key=lambda x: x[2], reverse=True)[:4]
                centers_final = [(int(round(c[0])), int(round(c[1]))) for c in circleshough_sorted]
        
        if len(centers_final) >= 4:
            # once again sort to 4, redundant i know
            centersorted_y = sorted(centers_final[:4], key=lambda p: p[1])
            toptwo = sorted(centersorted_y[:2], key=lambda p: p[0])
            bottomtwo = sorted(centersorted_y[2:], key=lambda p: p[0])
            outlist = [toptwo[0], bottomtwo[0], toptwo[1], bottomtwo[1]]
            return outlist

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
            out_image[:, :, c] = np.where(valmask, channel, out_image[:, :, c])
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
        out_image = np.where(valmask, warped, out_image)
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
        paddshiz = np.pad(image_bw, pad_width=1, mode='constant', constant_values=0)
        # Convolve with Sobel filters
        # standard process here
        iiixxx = cv2.filter2D(paddshiz, -1, self.SOBEL_X)[1:-1, 1:-1]
        iiiyyy = cv2.filter2D(paddshiz, -1, self.SOBEL_Y)[1:-1, 1:-1]
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

        # sx2, sy2, sxsy = None, None, None
        iiixxx, iiiyyy, iiiixyy = self.gradients(image_bw)

        # grab the second moments
        iii2x = iiixxx * iiixxx
        iii2y = iiiyyy * iiiyyy
        iiixy = iiixxx * iiiyyy

        # pad the second moments
        padsiz = ksize // 2
        # add pads
        iii2x_padded = np.pad(iii2x, pad_width=padsiz, mode='constant', constant_values=0)
        iii2y_padded = np.pad(iii2y, pad_width=padsiz, mode='constant', constant_values=0)
        iiixy_padded = np.pad(iiixy, pad_width=padsiz, mode='constant', constant_values=0)  

        # convolve with the gaussian kernel
        gaussian_kernel = cv2.getGaussianKernel(ksize, sigma)
        gaussian_kernel_2d = gaussian_kernel @ gaussian_kernel.T
        # and filter the second moments
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


        raise NotImplementedError

        return R


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



        raise NotImplementedError

        return x, y


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
        # get the harris response map
        rrr = self.harris_response_map(image_bw)
        # get the top k interest points
        xxx, yyy = self.nms_maxpool(rrr, k, ksize=8)
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


        raise NotImplementedError

        return warped_img


    def output_mosaic(self, img_src, img_warped):
        '''
        Input -
        :param img_src: Image 1
        :param img_warped: Warped Image
        Output -
        :return: Output Image Mosiac
        '''


        raise NotImplementedError

        return im_mos_out




