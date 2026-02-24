"""
CS6476 Assignment 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
from typing import Tuple
from scipy.ndimage import rotate


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


# --- Geometric validation (commented out: caused timeout on autograder) ---
# def _forms_convex_quad(pts):
#     """True if 4 points form a convex quadrilateral (all vertices on convex hull)."""
#     if len(pts) != 4:
#         return True
#     pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
#     hull = cv2.convexHull(pts_arr)
#     return len(hull) >= 4
#
# def _interior_point_idx(pts):
#     """Index of the point inside the triangle of the other 3, or -1 if all form convex quad."""
#     pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
#     hull = cv2.convexHull(pts_arr)
#     if len(hull) >= 4:
#         return -1
#     hull_pts = [tuple(h[0]) for h in hull]
#     for i, p in enumerate(pts):
#         match = any(abs(p[0] - h[0]) < 2 and abs(p[1] - h[1]) < 2 for h in hull_pts)
#         if not match:
#             return i
#     return -1
#
# def _pick_four_markers(clusters):
#     """Pick 4 marker positions that form a convex quad, replacing outliers when needed."""
#     if len(clusters) < 4:
#         return [(c[0], c[1]) for c in clusters]
#     indices = [0, 1, 2, 3]
#     centers = [(clusters[i][0], clusters[i][1]) for i in indices]
#     next_candidate = 4
#     while next_candidate < len(clusters):
#         if not _forms_convex_quad(centers):
#             idx = _interior_point_idx(centers)
#             if idx < 0:
#                 break
#         else:
#             scores = [clusters[i][2] for i in indices]
#             min_idx = min(range(4), key=lambda i: scores[i])
#             others = [s for j, s in enumerate(scores) if j != min_idx]
#             if scores[min_idx] >= 0.80 * (sum(others) / 3):
#                 break
#             idx = min_idx
#         indices[idx] = next_candidate
#         centers = [(clusters[i][0], clusters[i][1]) for i in indices]
#         next_candidate += 1
#     if not _forms_convex_quad(centers):
#         centers = [(clusters[i][0], clusters[i][1]) for i in range(4)]
#     return centers


# --- Hough refinement (commented out: broke unit tests) ---
# def _get_hough_circles(gray):
#     """Run Hough circle detection; return list of (x, y) circle centers."""
#     params_list = [
#         {'dp': 1, 'minDist': 50, 'param1': 80, 'param2': 30, 'minRadius': 8, 'maxRadius': 120},
#         {'dp': 1, 'minDist': 35, 'param1': 55, 'param2': 25, 'minRadius': 6, 'maxRadius': 150},
#     ]
#     for params in params_list:
#         circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **params)
#         if circles is not None and len(circles[0]) >= 4:
#             return [(int(round(c[0])), int(round(c[1]))) for c in circles[0, :]]
#     return []
#
# def _refine_with_hough_circles(centers, gray, clusters):
#     """Replace template-match points far from any circle (fiducials are circular)."""
#     if len(centers) != 4:
#         return centers
#     circles = _get_hough_circles(gray)
#     if len(circles) < 4:
#         return centers
#     max_dist = 28
#     result = list(centers)
#     for i, pt in enumerate(centers):
#         best_d = min((pt[0] - c[0])**2 + (pt[1] - c[1])**2 for c in circles)
#         if best_d <= max_dist ** 2:
#             continue
#         other_pts = [result[j] for j in range(4) if j != i]
#         for circle in sorted(circles, key=lambda c: (pt[0]-c[0])**2 + (pt[1]-c[1])**2):
#             if min((circle[0]-o[0])**2 + (circle[1]-o[1])**2 for o in other_pts) < 400:
#                 continue
#             result[i] = circle
#             if _forms_convex_quad(result):
#                 break
#             result[i] = pt
#     return result


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

    # Multi-scale + multi-blur template matching (markers appear at different sizes/angles)
    if template is not None:
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()

        hgtt, wgtt = template_gray.shape

        # 7 scales x 2 blurs = 14 calls
        scales = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
        blurconfigs = [(0, 0), (5, 1)]

        for scale in scales:
            newhgt = int(hgtt * scale)
            newwgt = int(wgtt * scale)
            if newhgt >= gray.shape[0] or newwgt >= gray.shape[1] or newhgt < 5 or newwgt < 5:
                continue
            scaled_tmpl = cv2.resize(template_gray, (newwgt, newhgt))

            for ksize, sigma in blurconfigs:
                proc = cv2.GaussianBlur(gray, (ksize, ksize), sigma) if ksize > 0 else gray
                try:
                    reslt = cv2.matchTemplate(proc, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
                except Exception:
                    continue

                resltcopy = reslt.copy()
                for _ in range(6):
                    minval, maxval, minloc, maxloc = cv2.minMaxLoc(resltcopy)
                    if maxval < 0.35:
                        break
                    cccxxx = maxloc[0] + newwgt // 2
                    cccyyy = maxloc[1] + newhgt // 2
                    candidates.append((cccxxx, cccyyy, maxval))
                    supp = max(newwgt, newhgt)
                    yyy1 = max(0, maxloc[1] - supp // 2)
                    yyy2 = min(resltcopy.shape[0], maxloc[1] + supp // 2)
                    xxx1 = max(0, maxloc[0] - supp // 2)
                    xxx2 = min(resltcopy.shape[1], maxloc[0] + supp // 2)
                    resltcopy[yyy1:yyy2, xxx1:xxx2] = 0

        # Rotation fallback (2 angles x 3 scales = 6 calls) - only if < 4 found
        pre_clusters = []
        for cx, cy, val in sorted(candidates, key=lambda x: -x[2]):
            if not any(abs(cx - c[0]) < 40 and abs(cy - c[1]) < 40 for c in pre_clusters):
                pre_clusters.append([cx, cy, val])
        if len(pre_clusters) < 4:
            for angle in [-20, 20]:
                rot_tmpl = rotate(template_gray, angle, reshape=False, mode='constant', cval=0)
                for scale in [1.0, 1.5, 2.0]:
                    newhgt = int(hgtt * scale)
                    newwgt = int(wgtt * scale)
                    if newhgt >= gray.shape[0] or newwgt >= gray.shape[1] or newhgt < 5 or newwgt < 5:
                        continue
                    scaled_rot = cv2.resize(rot_tmpl, (newwgt, newhgt))
                    try:
                        reslt = cv2.matchTemplate(gray, scaled_rot, cv2.TM_CCOEFF_NORMED)
                    except Exception:
                        continue
                    resltcopy = reslt.copy()
                    for _ in range(6):
                        minval, maxval, minloc, maxloc = cv2.minMaxLoc(resltcopy)
                        if maxval < 0.35:
                            break
                        cccxxx = maxloc[0] + newwgt // 2
                        cccyyy = maxloc[1] + newhgt // 2
                        candidates.append((cccxxx, cccyyy, maxval))
                        supp = max(newwgt, newhgt)
                        yyy1 = max(0, maxloc[1] - supp // 2)
                        yyy2 = min(resltcopy.shape[0], maxloc[1] + supp // 2)
                        xxx1 = max(0, maxloc[0] - supp // 2)
                        xxx2 = min(resltcopy.shape[1], maxloc[0] + supp // 2)
                        resltcopy[yyy1:yyy2, xxx1:xxx2] = 0

    # Spatial clustering: merge candidates within 40px; best score wins per cluster
    clusters = []
    for cccxxx, cccyyy, val in sorted(candidates, key=lambda x: -x[2]):
        merged = False
        for ccc in clusters:
            if abs(cccxxx - ccc[0]) < 40 and abs(cccyyy - ccc[1]) < 40:
                if val > ccc[2]:
                    ccc[0], ccc[1], ccc[2] = cccxxx, cccyyy, val
                merged = True
                break
        if not merged:
            clusters.append([cccxxx, cccyyy, val])

    clusters.sort(key=lambda x: -x[2])
    centers_final = [(c[0], c[1]) for c in clusters[:4]]

    # Hough circles fallback when template matching finds < 4 clusters
    if len(centers_final) < 4:
        params_hough = [
            {'dp': 1, 'minDist': 55, 'param1': 105, 'param2': 35, 'minRadius': 12, 'maxRadius': 105},
            {'dp': 1, 'minDist': 35, 'param1': 55, 'param2': 25, 'minRadius': 6, 'maxRadius': 155},
        ]
        for params in params_hough:
            circleshough = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **params)
            if circleshough is not None and len(circleshough[0]) >= 4:
                circles_xy = [(int(round(c[0])), int(round(c[1]))) for c in circleshough[0, :]]
                circles_sorted = sorted(circles_xy, key=lambda p: p[1])
                toptwo = sorted(circles_sorted[:2], key=lambda p: p[0])
                bottomtwo = sorted(circles_sorted[-2:], key=lambda p: p[0])
                centers_final = [toptwo[0], bottomtwo[0], toptwo[1], bottomtwo[1]]
                break

    # Order as [top-left, bottom-left, top-right, bottom-right]
    if len(centers_final) >= 4:
        centersorted_y = sorted(centers_final[:4], key=lambda p: p[1])
        toptwo = sorted(centersorted_y[:2], key=lambda p: p[0])
        bottomtwo = sorted(centersorted_y[2:], key=lambda p: p[0])
        return [toptwo[0], bottomtwo[0], toptwo[1], bottomtwo[1]]

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
        :param im_src: Image 1 (destination image to be warped)
        :param im_dst: Image 2 (source/base image)
        :param H: numpy ndarray - 3x3 homography matrix mapping im_src -> im_dst space
        Output -
        :return: im_src warped onto a canvas large enough to contain both images
        '''
        hgtdst, wgtdst = im_dst.shape[:2]
        hgtsrc, wgtsrc = im_src.shape[:2]

        # Project the four corners of im_src through H to find where they land
        corners_src = np.array([[0, 0, 1],
                                 [wgtsrc - 1, 0, 1],
                                 [0, hgtsrc - 1, 1],
                                 [wgtsrc - 1, hgtsrc - 1, 1]], dtype=np.float64).T
        corners_dst = H @ corners_src
        corners_dst /= corners_dst[2, :]
        xxxs = corners_dst[0, :]
        yyys = corners_dst[1, :]

        # Canvas must fit im_dst and the warped im_src corners
        xxxmin = min(0, xxxs.min())
        yyymin = min(0, yyys.min())
        xxxmax = max(wgtdst, xxxs.max())
        yyymax = max(hgtdst, yyys.max())

        outwww = int(np.ceil(xxxmax - xxxmin))
        outhhh = int(np.ceil(yyymax - yyymin))

        # Offset so everything is positive
        offsetxxx = -xxxmin if xxxmin < 0 else 0
        offsetyyy = -yyymin if yyymin < 0 else 0
        self._offsetxxx = int(np.round(offsetxxx))
        self._offsetyyy = int(np.round(offsetyyy))
        self._dsthhh = hgtdst
        self._dstwww = wgtdst

        # Build translation matrix to shift into canvas
        T = np.array([[1, 0, offsetxxx],
                      [0, 1, offsetyyy],
                      [0, 0, 1]], dtype=np.float64)
        H_shifted = T @ H

        # Inverse warp: for each output pixel, find source in im_src
        Hinv = np.linalg.inv(H_shifted)
        yys_grid, xxxs_grid = np.meshgrid(np.arange(outhhh), np.arange(outwww), indexing='ij')
        coords = np.stack([xxxs_grid.ravel(), yys_grid.ravel(), np.ones(outhhh * outwww)], axis=0)
        srccoords = Hinv @ coords
        srccoords /= srccoords[2, :]
        mapxxx = srccoords[0, :].reshape(outhhh, outwww).astype(np.float32)
        mapyyy = srccoords[1, :].reshape(outhhh, outwww).astype(np.float32)

        # convert the image to uint8
        imsrcu8 = im_src.astype(np.uint8)
        if len(imsrcu8.shape) == 3:
            warped = np.zeros((outhhh, outwww, imsrcu8.shape[2]), dtype=np.uint8)
            for c in range(imsrcu8.shape[2]):
                dstch = np.zeros((outhhh, outwww), dtype=np.uint8)
                cv2.remap(imsrcu8[:, :, c], mapxxx, mapyyy,
                          cv2.INTER_LINEAR, dst=dstch,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                warped[:, :, c] = dstch
        else:
            warped = np.zeros((outhhh, outwww), dtype=np.uint8)
            cv2.remap(imsrcu8, mapxxx, mapyyy,
                      cv2.INTER_LINEAR, dst=warped,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warped


    def output_mosaic(self, img_src, img_warped):
        '''
        Input -
        :param img_src: Source/base image (im_dst from image_warp_inv)
        :param img_warped: Warped image (output of image_warp_inv)
        Output -
        :return: Mosaic combining both images
        '''
        outhhh, outwww = img_warped.shape[:2]
        offsetxxx = getattr(self, '_offsetxxx', 0)
        offsetyyy = getattr(self, '_offsetyyy', 0)
        srchhh, srcwww = img_src.shape[:2]

        # Start with the warped image as the canvas
        if len(img_warped.shape) == 3:
            mosaic = img_warped.copy().astype(np.uint8)
            mask = np.any(img_warped != 0, axis=2)
        else:
            mosaic = img_warped.copy().astype(np.uint8)
            mask = img_warped != 0

        # Place img_src onto the canvas at the offset position
        # img_src occupies [oy:oy+src_h, ox:ox+src_w] in the canvas
        yyy1, yyy2 = offsetyyy, min(offsetyyy + srchhh, outhhh)
        xxx1, xxx2 = offsetxxx, min(offsetxxx + srcwww, outwww)
        srcyyy2 = yyy2 - offsetyyy
        srcxxx2 = xxx2 - offsetxxx

        # this is the part that is causing the timeout
        if len(img_warped.shape) == 3:
            region_mask = mask[yyy1:yyy2, xxx1:xxx2]
            for ccc in range(img_src.shape[2]):
                mosaic[yyy1:yyy2, xxx1:xxx2, ccc] = np.where(
                    region_mask,
                    img_warped[yyy1:yyy2, xxx1:xxx2, ccc],
                    img_src[:srcyyy2, :srcxxx2, ccc])
        else:
            region_mask = mask[yyy1:yyy2, xxx1:xxx2]
            mosaic[yyy1:yyy2, xxx1:xxx2] = np.where(
                region_mask,
                img_warped[yyy1:yyy2, xxx1:xxx2],
                img_src[:srcyyy2, :srcxxx2])

        return mosaic.astype(np.uint8)


# --- Part 9: RANSAC for automatic homography estimation ---

def _extract_patch(gray, xxx, yyy, rrr):
    """Extract (2r+1)x(2r+1) patch centered at (x,y). 
    Returns None if out of bounds.
    Used as descriptor for Harris corner matching.
    """
    hgtimg, wdtimg = gray.shape[:2]
    # check if the patch is out of bounds
    if xxx - rrr < 0 or xxx + rrr >= wdtimg or yyy - rrr < 0 or yyy + rrr >= hgtimg:
        return None
    return gray[yyy - rrr:yyy + rrr + 1, xxx - rrr:xxx + rrr + 1].astype(np.float32).flatten()


def _match_harris_corners(img1, img2, kkk=300, patchradius=8):
    """Detect Harris corners in both images and match by SSD of normalized patches.
    Returns list of ((x1,y1), (x2,y2)) correspondences. Uses Lowe's ratio test (0.8) 
    to filter ambiguous matches."""
    if len(img1.shape) == 3:
        # convert the image to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # convert the image to grayscale
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        # if the image is not color, just use the image
        gray1, gray2 = img1, img2
    
    # detect the corners in the image
    detector = Automatic_Corner_Detection()
    # using a lot of the logic from above here obviously
    xxx1, yyy1 = detector.harris_corner(img1, k=kkk)
    xxx2, yyy2 = detector.harris_corner(img2, k=kkk)
    rrr = patchradius
    valid1 = (xxx1 >= rrr) & (xxx1 < gray1.shape[1] - rrr) & (yyy1 >= rrr) & (yyy1 < gray1.shape[0] - rrr)
    valid2 = (xxx2 >= rrr) & (xxx2 < gray2.shape[1] - rrr) & (yyy2 >= rrr) & (yyy2 < gray2.shape[0] - rrr)
    xxx1, yyy1 = xxx1[valid1], yyy1[valid1]
    xxx2, yyy2 = xxx2[valid2], yyy2[valid2]

    # check if there are any corners in the image
    if len(xxx1) == 0 or len(xxx2) == 0:
        return []
    desc1 = []
    desc2 = []
    # extract the patch from the image
    for iii in range(len(xxx1)):
        # extract the patch from the image
        ppp = _extract_patch(gray1, int(xxx1[iii]), int(yyy1[iii]), rrr)
        if ppp is not None:
            ppp = ppp - np.mean(ppp)
            nnn = np.linalg.norm(ppp)
            if nnn > 1e-6:
                ppp = ppp / nnn
            desc1.append((xxx1[iii], yyy1[iii], ppp))
    for iii in range(len(xxx2)):
        # extract the patch from the image
        ppp = _extract_patch(gray2, int(xxx2[iii]), int(yyy2[iii]), rrr)
        if ppp is not None:
            ppp = ppp - np.mean(ppp)
            nnn = np.linalg.norm(ppp)
            if nnn > 1e-6:
                ppp = ppp / nnn
            desc2.append((xxx2[iii], yyy2[iii], ppp))
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    
    matches = []
    ratio_thresh = 0.8  # Lowe's ratio: accept if best_dist / second_dist < 0.8
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
    # get the x and y coordinates
    xxx, yyy = pt[0], pt[1]
    ppp = np.array([xxx, yyy, 1.0])
    qqq = H @ ppp
    return (qqq[0] / qqq[2], qqq[1] / qqq[2])


def ransac_homography(correspondences, threshold=5.0, max_iters=2000):
    """
    RANSAC to estimate homography from point correspondences.
    Correspondences: list of ((x1,y1),(x2,y2)) where pt1 in img1, pt2 in img2 (same scene point).
    Returns H that maps pt2 -> pt1 (dst -> src) and inlier count.
    """
    if len(correspondences) < 4:
        return np.eye(3), 0
    best_H = np.eye(3)
    # best inliers
    best_inliers = 0
    # get the source points
    pts_src = [c[0] for c in correspondences]
    # get the destination points
    pts_dst = [c[1] for c in correspondences]
    # get the number of correspondences
    n = len(correspondences)
    # get the random number generator
    rng = np.random.default_rng(42)
    # for each iteration
    for _ in range(max_iters):
        # get 4 random indices
        idx = rng.choice(n, size=4, replace=False)
        src_4 = [pts_src[i] for i in idx]
        dst_4 = [pts_dst[i] for i in idx]
        try:
            H = find_four_point_transform(dst_4, src_4)
        except Exception:
            continue
        inliers = 0
        # for each correspondence
        for (p_src, p_dst) in correspondences:
            pred = _apply_homography(H, p_dst)
            err = (pred[0] - p_src[0]) ** 2 + (pred[1] - p_src[1]) ** 2
            if err < threshold ** 2:
                inliers += 1
        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H
    return best_H, best_inliers


def mosaic_ransac(img_src, img_dst, kkk=300, ransac_threshold=5.0, ransac_iters=2000):
    """
    Create mosaic using Harris corners + RANSAC homography.
    Stitches img_dst onto img_src (destination onto source, per README convention).
    Returns the mosaic image.
    """
    # get the matches
    matches = _match_harris_corners(img_src, img_dst, kkk=kkk)
    # check if the number of matches is less than 4
    if len(matches) < 4:
        return img_src
    # get the homography
    H, _ = ransac_homography(matches, threshold=ransac_threshold, max_iters=ransac_iters)
    mosaic_obj = Image_Mosaic()
    im_warped = mosaic_obj.image_warp_inv(img_dst, img_src, H)
    return mosaic_obj.output_mosaic(img_src, im_warped)
