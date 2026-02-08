# Problem Set 2: Template Matching and FFT

## NOTE: Please refer to the PDF file for instructions. GitHub markdown does not render formulae and algorithms correctly. The information is identical between the markdown and PDF, but the PDF renders more cleanly.

# Assignment Description

## Description
Problem Set 2 is aimed at introducing basic building blocks of image
processing. Key areas that we wish to see you implement are: loading and
manipulating images, producing some valued output of images, and
comprehension of the structural and semantic aspects of what makes an
image. For this and future assignments, we will give you a general
description of the problem. It is up to the student to think about and
implement a solution to the problem using what you have learned from the
lectures and readings. You will also be expected to write a report on
your approach and lessons learned.

## Learning Objectives

-   Use Hough tools to search and find lines and circles in an image.

-   Use the results from the Hough algorithms to identify basic shapes.

-   Use template matching to identify shapes

-   Understand the Fourier Transform and its applications to images

-   Address the presence of distortion / noise in an image.

## Problem Overview

### Rules

You may use image processing functions to find color channels and load
images. Don't forget that those have a variety of parameters and you may
need to experiment with them. There are certain functions that may not
be allowed and are specified in the assignment's autograder Ed post.\
Refer to this problem set's autograder post for a list of banned
function calls.\
**Please do not use absolute paths in your submission code. All paths
should be relative to the submission directory. Any submissions with
absolute paths are in danger of receiving a penalty!**

### Files to submit
-   ps2.py
-   experiment.py


# Instructions

## Obtaining the Starter Files:

Obtain the starter code from GitHub

## Programming Instructions

Your main programming task is to complete the api described in the file
**ps2.py**. The driver program **experiment.py** helps to illustrate the
intended use and will output the files needed for the writeup.
Additionally there is a file **ps2_test.py** that you can use to test
your implementation. You can refer to the FAQ for a non-exhaustive list of banned functions.

## Write-up Instructions

Create **ps2_report.pdf** - a PDF file that shows all your output for
the problem set, including images labeled appropriately (by filename,
e.g. ps2-1-a-1.png) so it is clear which section they are for and the
small number of written responses necessary to answer some of the
questions (as indicated). For a guide as to how to showcase your
results, please refer to the latex template for PS2.

## How to Submit

Two assignments have been created on Gradescope: one for the report -
**PS2_report**, and the other for the code - **PS2_code** where you need
to submit ps2.py and experiment.py.

# 1. Hough Transforms \[10 points\]

## 1.a Traffic Light

First off, you are given a generic traffic light to detect from a scene.
For the sake of the problem, assume that traffic lights are shown as
below: (with red, yellow, and green) lights that are vertically stacked.
You may also assume that there is no occlusion of the traffic light.

![image](/Figures/Fig3.png)

It is your goal to find a way to determine the state of each traffic
light and position in a scene. Position is measured from the center of
the traffic light. Given that this image presents symmetry, the position
of the traffic light matches the center of the yellow circle.\
Complete your python ps2.py such that traffic_light_detection returns
the traffic light center coordinates (x, y) ie (col, row) and the color
of the light that is activated ('red', 'yellow', or 'green'). Read the
function description for more details.\
Testing:\
A traffic light scene that we will test will be randomly generated, like
in the following pictures and examples in the github repo.

![image](/Figures/Fig2.png)

Functional assumptions:\
For the sake of simplicity, we are using a basic color scheme, but
assume that the scene may have different color objects and backgrounds
\[relevant for part 2 and 3\]. The shape of the traffic light will not
change, nor will the size of the individual lights relative to the
traffic light. Size range of the lights can be reasonably expected to be
between 10-30 pixels in radius. There will only be one traffic light per
scene, but its size and location will be generated at random (that is, a
traffic light could appear in the sky or in the road--no assumptions
should be made as to its logical position). While the traffic light will
not be occluded, the objects in the background may be.

## Code:

Complete traffic_light_detection(img_in, radii_range)

You are allowed to use Hough transform tools.\
cv2.matchTemplate is NOT allowed.

## Report:

Place the coordinates using c2.putText before saving the output images.
Input: scene_tl_1.png. Output: ps2-1-a-1.jpg \[5\]

## 1.b Construction sign one per scene \[5 points\]

Now that you have detected a basic traffic light, see if you can detect
road signs. Below is the construction sign that you would see in the
United States (apologies to those outside the United States).\
Implement a way to recognize the signs:

![image](/Figures/Fig1.png)

Similar to the traffic light, you are tasked with detecting the sign in
a scene and finding the (x, y) i.e (col, row) coordinates that represent
the **polygon's centroid**.\
Functional assumptions:\
Like above, assume that the scene may have different color objects and
backgrounds. The size and location of the traffic sign will be generated
at random. While the traffic signs will not be occluded, objects in the
background may be.

## Code:
Complete the following functions. Read their documentation in ps2.py for
more details.

-   construction_sign_detection(img_in)

## Report:

Place the coordinates using c2.putText before saving the output images.
Input: scene_constr_1.png. Output: ps2-1-b-1.jpg \[5\]

# 2 Template Matching \[30 points\]

Template matching is a common image processing technique that is used to
find small parts of an image that match with the template image. In this
section, we will learn how to perform template matching using different
metrics to establish matching.\
We will try to retrieve the traffic light and the traffic sign in Part
1, by using their templates. In addition, We have the image of a Waldo
and another image where Waldo is hidden.

![image](/Figures/waldo_1.png)

We will attempt to find Waldo by matching the Waldo template with the
image using the following techniques:

-   Sum of squared differences: tm_ssd

-   Normalized sum of squared differences: tm_nssd

-   Normalized correlation: tm_nccor

We use the sliding window technique for matching the template. As we
slide our template pixel by pixel, we calculate the similarity between
the image window and the template and store this result in the top left
pixel of the result. The location with maximum similarity is then touted
a match for the template.

## Code:

Complete the template matching function. Each method is called for a
different metric to determine the degree to which the template matches
the original image. You'll be testing on the traffic signs used in Part
1, and Suggestion : For loops in python are notoriously slow. Can we
find a vectorized solution to make it faster?
Note: cv2.matchTemplate() isn't allowed.

## Report:

Pick the best of the 4 methods to display in the report.\
Input: scene_tl_1.png. Output: ps2-2-a-1.jpg \[5\]\
Input: scene_constr_1.png. Output: ps2-2-b-1.jpg \[5\]\
Input: waldo1.png. Output: ps2-2-c-1.jpg \[5\]

## Text:

2.d What are the disadvantages of using Hough based methods in finding
Waldo? Can template matching be generalised to all images? Explain
Why/Why not. Which method consistently performed the best, why? \[15\]\

# 3 Fourier Transform

In this section we will use the Fourier Transform to compress an image.
The Fourier transform is an integral signal processing tool used in a
variety of domains and converts a signal into individual spectral
components (sine and cosine waves). Another way of thinking about this
is that it converts a signal from the time domain to the frequency
domain. While signals like audio are a 1-dimensional signal, we will
apply the Fourier transform to images as a 2-dimensional signal. For
more information on the Fourier Transform, lectures 2C-L1 and 2C-L2
provide a good overview.

## 1-Dimensional Fourier Transform

The Fourier transform can be computed in two different algorithmic ways:
Discrete and Fast. In big O notation, the Discrete Fourier Transform
(DFT) can be computed in O($n^2$) time while the Fast Fourier Transform
can be computed in O($n \log(n))$ time. In this assignment, we will
implement the Discrete Fourier Transform.

The Discrete Fourier Transform can be defined as

![image](/Figures/DFT_1.png)

One way to calculate the Fourier Transform is a dot product between a coefficient matrix and the signal. Given a signal of length n, we define the coefficient matrix (n × n) as 

![image](/Figures/DFT_2.png)

where $j$ represents each row and $w = e^{-i2\pi/N}$. The vector resulting from $M_n(w) * f(x)$ is now your Fourier transformed signal! To compute the inverse of the fourier transform, $w$ is $e^{i2\pi/N}$ and $f(x)= \frac{1}{N} M_n(w)·F(x)$.

## Code:

Complete the following functions following the process above. Numpy
matrix operations can be used to simplify the calculation but np.fft
functions are not allowed in this section.

-   dft(x)

-   idft(x)

**Report:** No writeup for this section

## 2-Dimensional Fourier Transform

Now that we have computed the Fourier Transform for a 1-dimensional
signal, we can do the same thing for a 2-dimensional image. Remember
that the 2-dimensional Fourier transform simply applies the
one-dimensional transform on each row and column.

## Code:
You may use the functions from the last sections but np.fft functions are not allowed.

-   dft2(x)

-   idft2(x)

**Report:** No writeup for this section

# 4 Using the Fourier Transform for Compression \[15 points\]

Compression is a useful tool in all types of signal processing but
especially useful for images. Lets say you take a picture of your dog
that is 10 mb but want to reduce the file size and yet still maintain
the quality of the image. One method of doing this is to convert the
image into the frequency domain and keeping only the most dominant
frequencies. For this section, we will implement image compression and
pseudocode for the algorithm is shown below.

![image](/Figures/Compression.png)

To visualize the masked frequency image, be sure to shift all the low
frequencies to the center of the image with np.fft.fftshift.
Additionally, take $20*log(abs(x))$. This is done to properly visualize
the frequency domain image.

## Code:

-   compress_image_fft(x)

This functions are used in: compression_runner()

**Report:**

Use threshold percentages of 0.1, 0.05, and 0.001. Display both the
resulting image and the frequency domain image in the report for each.\
Outputs: ps2-4-a-1.jpg, ps2-4-a-2.jpg, ps2-4-a-3.jpg \[15 points, 5
each\]

# 5 Filtering with the Fourier Transform \[35 points\]

Now that we have seen how the Fourier Transform can be used for
compression, we will now use the Fourier Transform as a low-pass filter.
A low-pass filter similarly keeps all the low frequencies within the
image and 0's out all the high frequency components. We will follow the
process shown in lecture video 2C-L1-14 by first converting the image to
the frequency domain, masking the spectral image with a circle of radius
r, and converting the image back to pixel color values. The pseudocode
for the algorithm is given below:

![image](/Figures/LowPassFilter.png)

## Code:
-   low_pass_filter(img_bgr, r)

This function is used in: low_pass_filter_runner()

**Report:** Use radii of 100, 50, and 10. Display both the resulting
image and the frequency domain image in the report for each. Apply
$20*log(abs(x))$ to properly visualize the frequency domain image.\
Outputs: ps2-5-a-1.jpg, ps2-5-a-2.jpg, ps2-5-a-3.jpg \[15 points, 5
each\]

5-b What are the differences between compression and filtering? How does
this change the resulting image? \[10 points\]

5-c Given an image corrupted with salt and pepper pepper noise, what filtering method can effectively reduce/remove this noise? Also explain your choice of filtering method. \[10 points\]

## References
[1] The Waldo images for template matching were taken from this repo: https://github.com/vc1492a/Hey-Waldo.

# FAQ

Please check for your questions here before asking below. We will keep updating as we see new repeated questions.

#### Implementation:

Q: What are the OpenCV methods that we can use for this assignment? \
A: The objective of this assignment is for you to familiarize yourselves with Hough tools, template matching, and Fourier Transform. You can also use some other techniques to pre-process, or even after detection.
You can use the Hough tools in OpenCV, i.e houghLines, houghLinesP, houghCircles, etc.
You can use basic image processing functions of OpenCV or numpy for filtering, blurring, masking, finding edges, etc., These include but are not limited to cv2.dilate, bilateralFilter, fastNlMeansDenoisingColored, cv.Canny.
Other simple methods in Python standard library, numpy, or OpenCV can be used as long as the code passes the autograder.

Q: What are the allowed import packages? \
A:"numpy", "scipy", "cv2", "scipy", "matplotlib", "math", "itertools", "collections", "random", "enum"

Q: What OpenCV methods are NOT allowed in this assignment? \
A: Any object or shape detecting methods are not allowed.
These include but are not limited to cv2.findContours, cv2.SimpleBlobDetectors (which uses the cv2.findContours internally), cv.cornerHarris, cv2.connectedComponents, cv2.approxPolyDP, cv2.drawContours.

Q: Can I modify the signatures of Python methods in my code? \
A: Yes, as long as your function passes the autograder.

Q: What are the requirements for drawing the results on images? \
A: Make sure that the text and the mark are visible. Also, ensure that all text is readable for grading.

Q: Besides the functionality tests of Autograder, are there any other requirements for my code? \
A: We will check the code, and see if you have hardcoded the sign coordinates and the sign names or not. Clear names of functions and variables, and some comments are much appreciated but this is not mandatory and should not impact grading. Wherever there is duplication, try making a single function for it.

Q: My code complains about attribute error of OpenCV methods, how can I fix it? \
A: Make sure you are using the right names of methods or attributes for version 3.
Note: https://stackoverflow.com/questions/35609719/opencv-houghlinesp-parameters

Q: Can we assume each of the "on" colors are going to be exactly the same for all cases?\
A: Yes

Q: Can we assume traffic lights and signs are always in the upright position?\
A: Yes

Q: In the report, Traffic sign detection has "color". What should it be? \
A: You may ignore the color part for traffic signs. NOTE: Traffic lights still need to have the color component.

Q: What coordinates do we return for the traffic lights: \
A: The center of the traffic light, not the "on" colour.

Q: Can I change the template? \
A: Yes, currently the template has white spaces. Even when the template is used, you NEED to make sure that the bounding box is tightly wrapped around the sign in the test image. 

#### Grading:

Q: How will this assignment be graded? \
A: Report and code will account for 200 points.
Points breakdown:
- coding section: 110 points.
- report section: 90 points.

Q: Can I write (y,x) in my report instead of (x,y), and do I have to follow the format in the report? \
A: Yes, you’ll lose points for not following the format. The format is (x, y) ie (col, row)

Q: Is there a format for the text and drawing the center? \
A: No, as long as it is clearly visible to the TAs. For text, students have used white text with a black outline in the past, which looked very neat.

Q: What parts of the test cases are randomly generated? What’s tolerance? \
A: The traffic light can be in any position, and of varying sizes. The background of all scenes will remain unchanged.

Q: My script passes the Autograder sometimes and other times, it fails. What should I do? \
A: Test cases are randomly generated by Autograder. So if it fails, it only means that your code didn't give results within the tolerance.
If you have passed all the cases, then stop. Do not keep submitting it (unless you really want to make your function really robust, which is appreciated). We take the last submitted score, and not your best score.

Q: My report submission failed due to the file size limit, what should I do? \
A: Try to reduce the size of your report pdf using tools like https://smallpdf.com/compress-pdf.

Q: My submission returned an error message of “exceeded the timeout of -1 seconds”, what should I do? \
A: If you are getting this error, it means that you are submitting your work after the deadline of the problem set. It won't be recorded as your last submission, and hence, not be graded.

Q: Canvas has a due date and an available date, which one should I use? \
A: You should always use the due date.
The “due date” is the date after which it can no longer be turned in, while the “available until” date indicates the time after which it can be no longer downloaded.
Please refer to the course syllabus for instructions on schedule, time zone requirements, and late policy.

Q: Are we allowed to discuss our programs or specific implementations that we used for PS2? \
A: Discussions are fine, as long as you do not share the code.
If you want suggestions on the for loop and such, you could probably post pseudo-code.
Students can still drop this course and retake it at some point, and hence we do not allow code sharing in this course.

Q: Any general advice for completing the PS2? \
A: Start early (preferably today) and discuss more on ed.
The course is quickly picking up speed starting from PS2.
Many students in the past semesters spent non-trivial effort in the implementation, parameter tuning, and debugging.

Q: Can you share some methods or tricks for post-processing the results returned from the Hough methods? \
A: You can try the kmeans clustering method in OpenCV.

Q: How to tune parameters for Hough Circles? Which cause the most impact? \
A: dp: trial and error \
minDist: seems like it could be related to the current radius you are testing \
param1: has to do with the Canny method, so I separately tried canny with a few combinations of parameters until I got one that showed all three circles \
param2: trial and error \
minRadius: seems related to the current radius you are testing \
maxRadius: seems related to the current radius you are testing \
Yes, if you understand what each parameter does, you'd have to tune at max 3 parameters. (Which again, after 2 or 3 trials, you should get a fair idea how changing those parameters affect your output) \

Q: I am not able to detect all the lines and circles. \
A: Hough tools in openCV are very parameter-dependent. Tune all parameters, most importantly param1 and param2.

Q: Is there an easy way to tune parameters? \
A: yes, check opencv’s trackbar, or this. A former student has worked on a trackbar and was willing to share it with you all. https://github.gatech.edu/bkerner3/trackbar

Q: I don't see a test on gradescope for the traffic light or the construction sign. \
A: Grading for this part is manual.

Q: Tolerance for traffic light centers. \
A: 5 pixels in each of the x and y directions. For example, if the center is (100, 100), a result of (95, 105) is acceptable.

#### Notes:

1. Data post-processing
- Try first to cluster the data, then compute the geometric information to find/locate shapes
- For example, you can cluster by angles of lines, or centers of line segments
- Be prepared for edge cases, which can be improved with trial and error

2. OpenCV methods usage 
- For the `dp` parameter of `cv2.HoughCircles`, float values may perform differently (and better) than integer values

3. Hough methods tuning 
- Make sure you understand the meaning and effects of each individual parameter
- Experiment with one parameter at a time, trackbars can make it easier

4. Geometric computation 
- A trick to reduce complexity is to use the fact that the signs won't rotate (may not always be true for real-world images)
- Perform clustering on the lines with certain angles, 45 degrees for the squares, 30/60 degrees for triangles, etc.
