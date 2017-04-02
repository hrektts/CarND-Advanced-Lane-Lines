# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration2.jpg "Original"
[image2]: ./output_images/calibration2.jpg "Undistorted"
[image3]: ./output_images/test1_undist.jpg "Road Transformed"
[image4]: ./output_images/test1_binary.jpg "Binary Example"
[image5]: ./output_images/test1_src.jpg "Source points"
[image6]: ./output_images/test1_dst.jpg "Destination points"
[image7]: ./output_images/histogram.png "Histogram"
[image8]: ./output_images/window_search.png "Window search"
[image9]: ./output_images/test1_lined.jpg "Identified line"
[image10]: ./output_images/test1_result.jpg "Result"
[video1]: ./project_video.mp4 "Video"


## Camera Calibration

The code for my camera calibrationthis is in a function called `calibrate()`, which appears in [lines 177 through 234](./src/find_lane.py#L177-L234) of the file called `find_lane.py`, and an another function called `undistort()`, which appears in [lines 263 through 293](./src/find_lane.py#L263-L293) of the same file.

I started by preparing "object points", which was the (x, y, z) coordinates of the chessboard corners in the world. Here I assumed the chessboard was fixed on the (x, y) plane at z=0, such that the object points were the same for each calibration image.  Thus, `objpoint` was just a replicated array of coordinates, and `objpoints` was appended with a copy of it every time I successfully detected all chessboard corners in a test image.  `imgpoints` was appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

| Original image     | Corrected image    |
|:------------------:|:------------------:|
|![alt text][image1] |![alt text][image2] |

## Pipeline (single images)

To demonstrate this step, I will describe how I apply the image processing to one of the test images like this one:

![alt text][image3]

I used a combination of gradient and color thresholds to generate a binary image (thresholding steps at [lines 347 through 361](./src/find_lane.py#L347-L361) in `find_lane.py`).  Here's an example of my output for this step.

![alt text][image4]

The gradient threshold was applied for a gradient image that was applied Sobel operator following conversion to grayscale, in order to find edges in vertical direction.  The color thresholds were applied to the hue and the saturation channel of the image that was converted HLS color space in advance.  The first thresholds was effective because the hue of the lines was different from its surroundings and the second one was also effective because the saturation of the lines was relatively higher than its surroundings.

The code for my perspective transform appears in lines 363 through 379 in the file `find_lane.py`.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[img_size[0] / 2 - img_size[0] * 0.048, img_size[1] / 2 + img_size[1] * 0.14],
     [img_size[0] / 2 - img_size[0] * 0.28, img_size[1] - 50],
     [img_size[0] / 2 + img_size[0] * 0.31, img_size[1] - 50],
     [img_size[0] / 2 + img_size[0] * 0.053, img_size[1] / 2 + img_size[1] * 0.14]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
     [(img_size[0] / 4), img_size[1]],
     [(img_size[0] * 3 / 4), img_size[1]],
     [(img_size[0] * 3 / 4), 0]])
```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 589, 461      | 320, 0        |
| 282, 670      | 320, 720      |
| 1037, 720     | 960, 720      |
| 708, 461      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

| Undistorted image with source points drawn | Warped result with dest. points drawn |
|:------------------------------------------:|:-------------------------------------:|
|![alt text][image5]                         |![alt text][image6]                    |


Then I used a histogram to look for the left and right lanes.  I used a horizontal line at the center in the vertical direction as histogram data.  The figure below shows the histogram, and the points where the value is high indicate existence of the lines.

![alt text][image7]

For my beginning of the search, I divided the image into two images vertically, and set the point that has highest value as my starting point in each image.  This step appears in [lines 503 through 513](./src/find_lane.py#L503-L513) of the file called `find_lane.py`.

Next, I divided the image into 12 sections horizontally, such that I found lines in each section using different settings.  In the first section, that located at the top of the image, I set rectangular area which center has the same x value as the starting point calculated above.  The mean of the position of the whole pixel inside the rectangular was set as new starting point for the next section.  Then I applied this process to all the sections.  These steps appear in [lines 515 through 566](./src/find_lane.py#L515-L566) of the file called `find_lane.py`.  Here is the schematic of the rectangles:

![alt text][image8]

Then I fit my lane lines with a 2nd order polynomial using all the points detected in the above steps. After the first polynominal fit, I did fit several times using new points detected around the lane lines because applying the fit only one time could not take me appropreate result. These steps appear in [lines 583 through 591](./src/find_lane.py#L583-L591) and [lines 418 through 438](./src/find_lane.py#L418-L438) of the file called `find_lane.py`, and the result is like this:

![alt text][image9]

Next, I calculated the radius of curvature of the lins.  I used Bourne's method as descrived in his website "Interactive Mathematics"[[1](#bourne)].  I did this in [lines 598 through 600](./src/find_lane.py#L598-L600) and [lines 411 through 416](./src/find_lane.py#L411-L416) in my code in `find_lane.py`.

Finally, I plotted back down onto the road such that the lane area is identified clearly.  I implemented this step in [lines 632 through 653](./src/find_lane.py#L632-L653) in my code in `find_lane.py`.  Here is an example of my result on a test image:

![alt text][image10]

---

## Pipeline (video)

Here's a [link to my video result on YouTube](https://youtu.be/WROS2aRtOn4).

---

## Discussion

I started this project from my baseline where I implemented the mechanism provided in the class material.  It worked almost fine however it sometime lost yellow line at where the load was rough, and the line detection was not stable so that the detected line moved around frame by frame.  To improve these situation, I applied a method per each.

For the first problem, I added threshold of hue channel of the image to detect yellow color.  This worked fine and yellow line could be find stably.

For the second problem, I used moving average for calculating polynominal fit of the lane line.  For the second problem, I used moving average for calculating polynominal fit of the lane lines.  Concretely, I held coordinates of the detected points during latest 5 steps, and used all of them for calculating polynominal fit for the line.  This also worked fine.  However, this might be a probrem if the car goes to twisty mountain road because moving average works as low-pass filter, and line detector might not follow sudden change of the line curvature.  Actually, my program has not worked well on `harder_challenge_video.mp4` yet.

In the future work, more robust line detection which is not affected by the lightness of the environment would be needed.  In addition to it, some sanity check mechanisms are also needed in order to recover from wrong line detection.


## Reference

- <a name="bourne">[1]
  Bourne, Murray. "[Radius of Curvature](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)." Interactive Mathematics. N.p., n.d. Web. 01 Apr. 2017.
