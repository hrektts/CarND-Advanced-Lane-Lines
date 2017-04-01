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
[image7]: ./output_images/test1_lined.jpg "Identified line"
[image8]: ./output_images/test1_result.jpg "Result"
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

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image7]

### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image8]

---

## Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
