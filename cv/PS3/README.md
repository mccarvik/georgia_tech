# Problem Set 3: Introduction to AR and Image Mosaic

# Assignment Description

## Description
Problem Set 3 introduces basic concepts behind Augmented Reality, using the contents that you will learn in modules 3A-3D and 4A-4C: Projective geometry, Corner detection, Perspective imaging, and Homographies, respectively.

Additionally, you will also learn how to insert images within images and stitch multiple images together.

## Learning Objectives

-  Find markers using corner detection and / or pattern recognition.
-  Learn how projective geometry can be used to transform a sample image from one plane to another.
-  Address the marker recognition problem when there is noise in the scene.
-  Implement backwards (reverse) warping.
-  Implement Harris corner detection to identify correspondence points for an image with multiple views.
-  Address the presence of distortion / noise in an image.


All tests in the autograder generate random scenes each time you submit your ps3.py script. Your functions should only return the information specified in each method's description located in ps3.py.

FAQs can be found at the bottom of this document.

# Problem Overview

## Methods to be Used

In this assignment you will use methods for Feature Correspondence and Corner Detection. You will also apply methods for Projective Geometry and Image Warping, however you will do these manually using linear algebra.

## Rules

You may use image processing functions to find color channels, load images, and find edges (such as with Canny). Don’t forget that those have a variety of parameters that you may need to experiment with. There are certain functions that may not be allowed and are specified in the problem descriptions and the FAQ at the bottom.

**Please do not use absolute paths in your submission code. All paths should be relative to the submission directory. The staff will not award points if you lose points on the autograder for using absolute paths!**

## Instructions

### Obtaining the Starter Files

Obtain the starter code from the PS3 GitHub repo.

### Programming Instructions

Your main programming task is to complete the API described in `ps3.py`.  The driver program `experiment.py` helps to illustrate the intended use and will output the files needed for the write-up.

### Write-Up Instructions

Create `ps3_report.pdf` - a PDF file that shows all your output for the problem set, including images labeled appropriately (by filename, e.g. `ps3-1-a-1.png`) so it is clear which section they are for, as well as a number of written responses necessary to answer some of the questions (as indicated). Please refer to the Latex template for PS3.

### How to Submit

Two assignments have been created on Gradescope. One for the report - `PS3_report`, and one for the code - `PS3_code`. 

* Report: the report (PDF only) must be submitted to the `PS3_report` assignment.
* Code: all files must be submitted to the `PS3_code` assignment. DO NOT upload zipped folders or any sub-folders, please upload each file individually. Drag and drop all files into Gradescope.

Note that your Gradescope submission is your **last** submission, not your best submission. If you need to revert to a previous submission you can access your previous submissions and download them via Gradescope.

## Notes

* You can only submit to the autograder **10 times in an hour**. You'll receive a message like `"You have exceeded the number of submissions in the last hour. Please wait for 36.0 mins before you submit again."` when you exceed those 10 submissions. You'll also receive a message `"You can submit 8 times in the next 53.0 mins"` with each submission so that you may keep track of your submissions.

* If you wish to modify the autograder functions, create a copy of those functions and DO NOT mess with the original function call.

**YOU MUST SUBMIT your report and code separately, i.e., two submissions for the code and the report, respectively. Only your last submission before the deadline will be counted for each of the code and the report**

## Write-up Instructions

The assignment will be graded out of 100 points. Only the last submission before the time limit will be considered. The code portion (autograder) represents **60%** of the grade and the report the remaining **40%**.

The images included in your report must be generated using experiment.py. This file should be set to run as-is to verify your results. **Your report grade will be affected if we cannot reproduce your output images.**

The report grade breakdown is shown in the question heading. As for the code grade, you will be able to see it in the console message you receive when submitting. The coding portion is out of 166 points (so 166/166 gets you the full 60% credit).

# Assignment Overview

A glass/windshield manufacturer wants to develop an interactive screen that can be used in cars and eyeglasses. They have partnered with a billboard manufacturer to render marketing products onto markers in the real world.

Their goal is to detect four points (markers) currently present in the screen’s field-of-view and insert an image or video in the scene. To help with this task, the advertising company is installing blank billboards with four distinct markers, which determine the area’s intended four corners. The advertising company plans to insert a target image/video into this space.

They have hired you to produce the necessary software to make this happen! They have set up their sensors so that you will receive an image/video feed and a target image/video. They expect an altered image/video that contains the target content rendered in the scene, visible on the screen.

## Part 1: Marker Detection in a Simulated Scene [40]

The first task is to identify the markers for this Augmented Reality exercise. In real practice, markers can be used (in the form of unique pictures) that stand out from the background of an image. Below is an image with four markers.

<img width="878" alt="Fig1" src="https://github.gatech.edu/OMSCS6476-Spring2025/PS3/assets/40258/9577912a-acf3-4230-9e3b-c084743e9b4b">

Notice that they contain a cross-section bounded by a circle. The cross-section is useful in that it forms a distinguished corner. In this section, you will create a function/set of functions that can detect these markers, as shown above. You will use the images provided to detect the (x, y) center coordinates of each of these markers in the image. The position should be represented by the center of the marker (where the cross-section is). To approach this problem you should consider using techniques like detecting circles in the image, detecting corners and/or detecting a template.

**Code**: Complete `find_markers(image)`

You will use the function `mark_location(image, pt)` in experiment.py to create a resulting image that highlights the center of each marker and overlays the marker coordinates in the image. You have lots of flexibility for how you approach this, as a starting point we recommend using template matching.

Each marker should present its location similar to this:

<img width="358" alt="Fig2" src="https://github.gatech.edu/OMSCS6476-Spring2025/PS3/assets/40258/44199555-15fd-45ee-a3ad-12c1f94d8983">

Images like the one above may not be that hard to solve. However, in a real-life scene, it proves to be much more difficult. Make sure your methods are robust enough to also locate the markers in images like the one below, where there could be other objects in the scene:

<img width="884" alt="Fig3" src="https://github.gatech.edu/OMSCS6476-Spring2025/PS3/assets/40258/9bfc9ca5-e17d-48c3-af3a-b6e7751f3a4a">

Let’s step it up. Now that you can detect markers on a blank background, assume there is “noise” in the scene (i.e. rain, fog, etc.). This helps ensure that our advertisements can be placed reliably in the scene.

<img width="888" alt="Fig4" src="https://github.gatech.edu/OMSCS6476-Spring2025/PS3/assets/40258/6c2feb39-9bc3-461d-a51d-f209d41d6bc8">

All tests in this part start by creating an image with a white background. Second, four markers are placed in random locations simulating the scenes that are present in the input_images directory.
- find_markers on empty background (similar to sim_clear_scene.jpg)
- find_markers with noise: just circles (similar to sim_noisy_scene_1.jpg)
- find_markers with noise: circles + gaussian (similar to sim_noisy_scene_2.jpg)

**Report**: This part will only be graded by the autograder. Do not include this part in your report.

## Part 2: Marker detection in a Real Scene [5]

Now that you have a working method to detect markers in simulated scenes, you will adapt it to identify these same markers in real scenes like the image shown below. Use the images provided to essentially repeat the task of section 1 above and draw a box (four 1-pixel wide lines, RED color) where the box corners touch the marker centers.

<img width="752" alt="Fig5" src="https://github.gatech.edu/OMSCS6476-Spring2025/PS3/assets/40258/75042598-48bb-4bd3-b593-253d585381e6">

**Code**: Complete `draw_box(image, markers)`

A blank image and four random marker points are generated. Your output should return just the rectangle perimeter with a line thickness of 1. The number of nonzero pixels in this image should be close to the euclidean distances of each rectangle side: 
  - ```dist(top_left, bottom_left) + dist(top_left, top_right) + dist(bottom_right, top_right) + dist(bottom_right, bottom_left)```

**Report**: This part will only be graded by the autograder. Do not include this part in your report.
 
### Part 3: Projective Geometry [60]

Now that you know where the billboard markers are located in the scene, we want to add the marketing image. The advertising company requires that their client’s billboard image is visible from all possible angles since you are not just driving straight into the advertisements. Unphased, you know enough about computer vision to introduce projective geometry. The next task will use the information obtained in the previous section to compute a transformation matrix `H`. This matrix will allow you to project a set of points (x, y) to another plane represented by the points (x’, y’) in a 2D view. In other words, we are looking at the following operation:

<img width="361" alt="Fig6" src="https://github.gatech.edu/OMSCS6476-Spring2025/PS3/assets/40258/75bf5101-572e-417c-9ee3-f0621739d121">

In this case, the 3x3 matrix is a *homography*, also known as a *perspective transform* or *projective transform*. There are eight unknowns, **a** through **h**, and **i** is 1. If we have four pairs of corresponding `(u,v) <==> (u',v')` points, we can solve for the homography.

The objective here is to insert an image in the rectangular area that the markers define. This insertion should be robust enough to support cases where the markers are not in an orthogonal plane from the point of view and present rotations. Here are two examples of what you should achieve:

<img width="989" alt="Fig7" src="https://github.gatech.edu/OMSCS6476-Spring2025/PS3/assets/40258/29b49da7-128b-4432-9850-8110a48d7358">

When implementing `project_imageA_onto_imageB()` you will have to make the design choice between forward or backward warping. To make the best choice, you should test both approaches and comment in the report on what helped you choose one method over the other. (Note: to better see differences between the two methods you should pick a marketing image with low resolution).

**Code**: Complete the following functions:
* `get_corners_list()`: Your output is checked to see if it returns the right type and complies the ordering specified in the ps3.py documentation.
* `find_four_point_transform(src_points, dst_points)`: Random points are generated and, from these, a reference transformation matrix H is calculated. Your output is used to transform the reference points and verify them with a reference solution using the matrix H.
* `project_imageA_onto_imageB(imageA, imageB, homography)`: Two random images are generated one with all zeros and the second one with a random gradient color configuration. The gradient image is then projected to the black image plane using a reference homography. Your output is then compared to a reference solution using the same similarity function provided in ps3_test.py.

**Report**: Report what warping technique you have used and comment on what led you to choose this method.

## Part 4: Finding Markers in a Video [35]

Static images are fine in theory, but the company wants this functional and put into practice. That means, finding markers in a moving scene.

In this part, you will work with a short video sequence of a similar scene. When processing videos, you will read the input file and obtain images (frames). Once the image is obtained, you will apply the same concept as explained in the previous sections. Unlike the static image, the input video will change in translation, rotation, and perspective. Additionally, there may be cases where a few markers are partially visible. Finally, you will assemble this collection of modified images into a new video. Your output must render each marker position relative to the current frame coordinates.

Besides making all the necessary modifications to make your code more robust, you will complete a function that outputs a video frame generator. This function is almost complete and it is placed so that you can learn how videos are read using OpenCV. Follow the instructions placed in ps3.py.

* First we will start with the following videos.
  * Input: **ps3-4-a.mp4**
  * Input: **ps3-4-b.mp4**
  * Output: **ps3-4-a-1.png, ps3-4-a-2.png, ps3-
4-a-3.png, ps3-4-a-4.png, ps3-4-a-5.png, ps3-
4-a-6.png**
* Now work with noisy videos:
  * Input: **ps3-4-c.mp4**
  * Input: **ps3-4-d.mp4**
  * Output: **ps3-4-b-1.png, ps3-4-b-2.png, ps3-
4-b-3.png, ps3-4-b-4.png, ps3-4-b-5.png, ps3-
4-b-6.png**

**Code**: Complete `video_frame_generator(filename)`: A video path is passed to this function. The output is then verified for type and shape. After this, the number of frames counted by repeatedly calling the next() function is compared to the original number of frames.

**Report**: Report the 3 keyframes per video in the report.

## Part 5: Final Augmented Reality [55]

Now that you have all the pieces, insert your advertisement into the video provided. Pick an image and insert it in the provided video.

* First we will start with the following videos.
  * Input: **ps3-4-a.mp4**
  * Input: **ps3-4-b.mp4**
* Now work with noisy videos:
  * Input: **ps3-4-c.mp4**
  * Input: **ps3-4-d.mp4** - Frames to record: 207, 367, and 737
  * Output: **ps3-5-b-4.png, ps3-5-b-5.png, ps3-
5-b-6.png**

**Report**: In order to grade your implementation, you should extract a few frames from your last generated video and add them to the corresponding slide in your report.

In the next few tasks, you will be reusing the tools that you have built to stitch together 2 images of the same object from different viewpoints to create a combined panorama. 

## Part 6: Finding Correspondence Points in an Image [10]

In this part of the project, you have to manually select correspondence points with mouse clicks from two views of the input image. The functions for this task will be provided to you in the class `Mouse_Click_Correspondence(object)`.
The points selected will have to be used to get the homography parameters. The sensitivity of the result would depend heavily on the accuracy of these correspondence points. Make sure to choose distinctive points in the image that are present in both the views. 

The functions in the class `Mouse_Click_Correspondence(object)` does not return anything and will create 2 numpy files (`p1.npy` and `p2.npy`) which will store the coordinates of the selected correspondence points. 

**Report**: In order to grade your implementation, attach a screenshot of the images with the manually selected points in the corresponding slide in the report.

* Image 1: **ps3-6-a-1**
* Image 2: **ps3-6-a-2**

## Part 7: Image Stitching I (Manual Mosaic) [30]

In this task, you will be completing the code to perform the final image stitching and create the output mosaic. So far, you have calculated the homography transform from one image to the other. Use perspective transformation to stitch the two images together.

**NOTE**: Ensure that you stitch (or attach) the destination image onto the source image and not the other way around. This is purely for the purpose of matching the convention on the autograder while evaluating your code.

**Code**: Complete the following function from the `Image_Mosaic()` class:
* `image_warp_inv()`
* `output_mosaic()`
Recall concepts from Projective Geometry section.

**Report**: Place the generated panorama (`image_mosaic_a`) into **ps3-9-1** (9: Image Stitching)

## Part 8: Automatic Correspondence Point Detection

In this task, instead of manually selecting the correspondence points, you will write code to automate this process. The inputs to this task are the two images and the output needs to be the homography matrix of the required transformation. Use Harris Corner Detection to perform this task. The implemented solution must be able to work with RGB images. You should refer to module 4A-L2 to learn about Harris corners.

**Code**: Complete the following functions under the `Automatic_Corner_Detection()` class:
* `gradients()`
* `second_moments()`
* `harris_response_map()`
* `nms_maxpool()`
* `harris_corner()`

**Report**: There is no Report section for this part.

## Part 9: Image Stitching II: (RANSAC) [20]

In this last section, you will implement the RANSAC algorithm to obtain the best matches among the detected corners using a feature detector (like Harris Corners). You may refer to Lecture 4C-L2 to understand how RANSAC works.

**Code**: There is no template provided for this task and you are free to implement it as creatively as you like. There is also no autograder component for this section. Add your code to the existing `ps3.py` file.

**Report**:
* Place the mosaic generated using RANSAC into **ps3-9-2**.
* Comment on the quality difference between the two outputs and how it relates to the importance of choosing the correct correspondence points for the image.

Well done!

# General Notes

We want to reiterate our advice about using Numpy indexing methods that can speed up your code. Nested for loops tend to be very slow in Python so try to avoid them whenever you can. Think about how you can implement these methods using linear algebra operations. This will be particularly useful for parts 3 - 6 in your report. However, be mindful that some matrix operations can overload the memory of the autograder, so be careful about the input dimensions of your matrix multiplications.

Remember that you should not post code on Piazza. Also, make sure you search the FAQ and Ed to see whether your question has been answered before posting.

There are certain openCV functions that are **not** allowed. So far this is the preliminary list of banned function calls:
```
cv2.findHomography
cv2.getPerspectiveTransform
cv2.findFundamentalMat
cv2.warpPerspective
cv2.goodFeaturesToTrack
cv2.warpAffine
```
Allowed imports functions can be found in FAQs.md

This list may change as we find more examples so when in doubt please ask. In the event that you are using these functions you will see a warning message in your submission. You will receive zero points if you use any of these functions in the assignment.

The autograder is set to timeout after 60 seconds, which is plenty of time for these small tests to complete (~20 seconds total). If you receive a timeout message it means you need to optimize your code. Additionally, Gradescope has a global quota that is max 10 submissions in a 1h window.

# FAQ

### Part 1:

Q: Some of the coordinates drawn on the images are outside the image. Am I supposed to fix it? Or should I just write down the coordinates in the report?

A: If the text outbounds the image in a few frames of a video you can leave it as it is. You can then include the coordinates in the report.


### Part 2:

Q: How far away can our vertices be from the center of the markers for part 2 of the assignment (marker detection in real scenes)?

A: The rectangle corners have to be in the marker areas.

Q: How should we handle the "ps3-2-c_base.jpg" image that is rotated 90 degrees in terms of "top left" or "bottom right"?

A: you can return the points in the requested order [top-left, bottom-left, top-right, bottom-right] as they appear in the image. And then draw the rectangle.

Q: Draw box errors:

A: Please check if [255, 255, 0] works, or [0, 0, 255].


### Part 3:
Q: It seems part 3, 4, 5 all depend on accurately identifying the markers. If one marker fail, will I still receive partial grades even if these questions do not ask for markers directly?

A: The corners should be in the marker area. Part2 through 5: -1 per missed corner. 

NOTE: Please make sure the the coordinates passed are in the right order.

### Part 4:
Q: For the video portion of the assignment, will it okay for "a few" frames to glitch out, or do we need to meet the "within-the-rectangles" requirement for 100% of the frames ?

A: Ideally, you are expected to meet the requirement for 100%. You won’t loose any points if one or two frames have are off. If we see a lot of them, then there will be a penalty.

### Part 5:
Q: Will the TAs be re-running our code for Part 5 and 6 (Augmented Reality Video)? My code works but it runs so slow (~30 mins for 1 video in part 5).

A: We do run the code. For those parts you should be able to optimize the code to run under 2min.

### Part 8:
There are two parts to the Part 8 Harris Corner Detection. You need both functions to pass the test in Autograder:
harris_response_map: Follow the algorithm in Module 4A-L2
Normalize the response map to a 0-1 range after getting it from harris_response_map, so the maximum confidence is 1.0 and easier to debug by converting to percentage.
nms_maxpool_numpy: 
- Suppress any point below the median
- maxpool in 2d with padding so that the output of the maxpool is same size as the non-padded Response Map. Use a ksize of 7.
- Filter out the Response Map to remove points that are not the same in maxpooled response map
- sort on confidences acquired from the filtered Response Map, get the top k values

Do not change the ransac_homography_matrix apart from the parameters p, s, e.

While using sobel filter and gaussian blur, make sure to add padding of 0's before doing the filtering.

Without padding with 0, blurring causes the edges of the image to have higher values causing corners to be detected on the edges of the image. I ran a few tests showing this effect and here are some results
"image":

```[[0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0.]] 
 ```

 Gaussian Blur without padding
 ```
 [[0.06651863 0.03675723 0.03903017 0.03981863 0.03903017 0.03675723
  0.06651863]
 [0.03675723 0.02031151 0.0215675  0.0220032  0.0215675  0.02031151
  0.03675723]
 [0.03903017 0.0215675  0.02290116 0.0233638  0.02290116 0.0215675
  0.03903017]
 [0.03981863 0.0220032  0.0233638  0.02383578 0.0233638  0.0220032
  0.03981863]
 [0.03903017 0.0215675  0.02290116 0.0233638  0.02290116 0.0215675
  0.03903017]
 [0.03675723 0.02031151 0.0215675  0.0220032  0.0215675  0.02031151
  0.03675723]
 [0.06651863 0.03675723 0.03903017 0.03981863 0.03903017 0.03675723
  0.06651863]]
```

You can see that the corners of the image have higher value than the centers leading to problems.
Gaussian Blur after Padding (padding removed afterward blurring) provides a much smoother blurring:
```[[0.01662966 0.01837862 0.01951509 0.01990932 0.01951509 0.01837862
  0.01662966]
 [0.01837862 0.02031151 0.0215675  0.0220032  0.0215675  0.02031151
  0.01837862]
 [0.01951509 0.0215675  0.02290116 0.0233638  0.02290116 0.0215675
  0.01951509]
 [0.01990932 0.0220032  0.0233638  0.02383578 0.0233638  0.0220032
  0.01990932]
 [0.01951509 0.0215675  0.02290116 0.0233638  0.02290116 0.0215675
  0.01951509]
 [0.01837862 0.02031151 0.0215675  0.0220032  0.0215675  0.02031151
  0.01837862]
 [0.01662966 0.01837862 0.01951509 0.01990932 0.01951509 0.01837862
  0.01662966]]
  ```

### Autograder:

Q: What packages are allowed to be imported?

A: ```"numpy", "scipy", "cv2", "scipy.ndimage", "scipy.ndimage.rotate", "matplotlib", "math", "itertools", "itertools.combinations", "collections", "random", "enum", "time"```

Q: What functions are allowed?

A: Few of the functions(that students have asked/used in the previous semesters) that are allowed are below.  
But understand these functions before you use them.
```cv2.matchTemplate, numpy.rot90, cv2.pyrDown(), cv2.remap, SimpleBlobDetector, all linalg functions, cv2.fillPoly, cv2.cornerEigenValsAndVecs, approxPolyDP, scipy.ndimage.rotate...```

If you use cv2.remap, be sure to use the keyward arg for destination, as assignment may not always work, ex. cv2.remap(...,dst=imageB_copy,...)

Q: What is the order for the corners?

A: ```[top-left, bottom-left, top-right, bottom-right]```

Q: What colour should I use for draw_box?

A: Red.

Q. What is the video grading criteria?

A: You need to be within 2 pixel tolerance at least 80% of the times. For the rest of the cases, it will be partially graded.

NOTE:
Below is the allowed imports for PS3
`"numpy", "scipy", "cv2", "scipy.ndimage", "scipy.stats", "scipy.ndimage.rotate",
"matplotlib", "math", "itertools", "itertools.combinations", "collections",
"random", "enum", "time", 
"scipy.cluster", "matplotlib.pyplot", "scipy.cluster.vq.kmeans",
"numpy.linalg", "operator.itemgetter", "scipy.optimize",
"scipy.signal", "cv2.filter2d",
"scipy.spatial.distance", "scipy.cluster.vq.kmeans2", "scipy.signal.convolve2d",
"scipy.ndimage.interpolation.rotate",
"PIL", "typing", "typing.Tuple", cv2.findContours
cv2.minEnclosingCircle
cv2.cornerHarris (for P9 ONLY)
`

What this means is that you can only do
```import xyz as abc```
Where `xyz` HAS to be from one of the functions above. 

Gradescope timeout: Please check if there is a function in your code that is taking very long to run and optimize it. Autograder runs your code on 5-10 samples, and timeout is set at 20 mins. 
