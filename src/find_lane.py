#!/usr/bin/env python
""" Advanced Lane Finding Project
"""

import logging
import getopt
import glob
import os
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(module)s: %(funcName)s: %(message)s'))
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    """ Applies Sobel operator and threshold to an image.

    Args:
        img: A single channel image to be processed.
        orient: The direction used to calculate the derivative. Takes 'x' or 'y'.
        thresh: Minimum and maximum thresholds used to binalize the processed image.

    Returns:
        A processed binary image.

    """
    # Applies x or y gradient and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """ Calculates gradient magnitude of an image and applies threshold.

    Args:
        img: A single channel image to be processed.
        sobel_kernel: The kernel size used by Sobel operator.
        thresh: Minimum and maximum thresholds used to binalize the processed image.

    Returns:
        A processed binary image.

    """
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    gradmag = np.uint8(255 * gradmag / np.max(gradmag))
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ Thresholds an image for a given range and Sobel kernel

    Args:
        img: A single channel image to be processed.
        sobel_kernel: The kernel size used by Sobel operator.
        thresh: Minimum and maximum thresholds used to binalize the processed image.

    Returns:
        A processed binary image.

    """
    # Takes the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculates the direction of the gradient
    absgrad = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Creates a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgrad)
    binary_output[(absgrad >= thresh[0]) & (absgrad <= thresh[1])] = 1

    return binary_output

def channel_thresh(img, thresh=(0, 255)):
    """ Calculates gradient magnitude of an image and applies threshold.

    Args:
        img: A single channel image to be processed.
        thresh: Minimum and maximum thresholds used to binalize the processed image.

    Returns:
        A processed binary image.

    """
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(img)
    binary_output[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary_output

def channel_and(img1, img2):
    """ Pixel-wise And operation.

    Args:
        img1: A single channel image.
        img2: An another single channel image.

    Returns:
        A processed binary image.

    """
    binary_output = np.zeros_like(img1)
    binary_output[(img1 == 1) & (img2 == 1)] = 1

    return binary_output

def channel_or(img1, img2):
    """ Pixel-wise Or operation.

    Args:
        img1: A single channel image.
        img2: An another single channel image.

    Returns:
        A processed binary image.

    """
    binary_output = np.zeros_like(img1)
    binary_output[(img1 == 1) | (img2 == 1)] = 1

    return binary_output

class ImageProcessor:
    """ A image processor
    """
    def __init__(self, data_dir, cal_img_dir, test_img_dir, out_img_dir):
        """ The initializer

        Args:
            data_dir: A directory used to load and save parameters.
            cal_img_dir: A directory where calibration images are read from.
            test_img_dir: A directory where test images are read from.
            out_img_dir: A directory where processed images are written.

        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        self.cal_img_dir = cal_img_dir
        if not os.path.exists(cal_img_dir):
            os.mkdir(cal_img_dir)

        self.test_img_dir = test_img_dir
        if not os.path.exists(test_img_dir):
            os.mkdir(test_img_dir)

        self.out_img_dir = out_img_dir
        if not os.path.exists(out_img_dir):
            os.mkdir(out_img_dir)

        self.cal_data_file = None
        self.img_shape = None
        self.cal_img_names = None
        self.camera_mtx = None
        self.dist_coeffs = None
        self.invmtx = None

    def read_cal_img_name(self, ext='jpg'):
        """ Reads all image names in the calibration image directory.

        Args:
            ext: The extention of the images

        """
        path = self.cal_img_dir + '/*.' + ext
        self.cal_img_names = []
        for name in glob.glob(path):
            self.cal_img_names.append(name)

    def read_cal_img_shape(self):
        """ Reads the shape of images.
        """
        logger.debug('chack the size of calibration images')

        if self.cal_img_names is None:
            self.read_cal_img_name()

        shapes = {}
        for name in self.cal_img_names:
            img = cv2.imread(name)
            if img.shape in shapes:
                shapes[img.shape] += 1
            else:
                shapes[img.shape] = 1

        self.img_shape = max(shapes, key=lambda k: shapes[k])

    def calibrate(self, pattern_size=(9, 6), cal_data_file='camera.p'):
        """ Calibrates camera using chessboard images.

        Args:
            pattern_size: The number of crossing points of chessboard image.
                          (x-direction, y-direction)
            cal_data_file: A file name used to store calibration data.

        Returns:
            Succeeded (False) or Failed (True)

        """
        logger.debug('starts camera calibration')

        if self.cal_img_names is None:
            self.read_cal_img_name()

        if self.img_shape is None:
            self.read_cal_img_shape()

        objpoint = np.zeros((np.prod(pattern_size), 3), np.float32)
        objpoint[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

        objpoints = []
        imgpoints = []
        for name in self.cal_img_names:
            img = cv2.imread(name)
            if self.img_shape != img.shape:
                logger.debug('detects different image size: %s', img.shape)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if not found:
                logger.debug('could not detect point(s) in %s', name)
                continue

            objpoints.append(objpoint)
            imgpoints.append(corners.reshape(-1, 2))

        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints,
                                                   imgpoints,
                                                   self.img_shape[0:2],
                                                   None,
                                                   None)
        if ret:
            self.camera_mtx = mtx
            self.dist_coeffs = dist

            self.cal_data_file = self.data_dir + '/' + cal_data_file
            camera = {'cameraMatrix': mtx, 'distCoeffs': dist}
            with open(self.cal_data_file, mode='wb') as f:
                pickle.dump(camera, f)
                return False
        else:
            logger.debug('calibration failed')
            return True

        return True

    def load_calibration_data(self, cal_data_file='camera.p'):
        """ Loads camera calibration data

        Args:
            cal_data_file: A file name used to read calibration data.

        Returns:
            Succeeded (False) or Failed (True)

        """
        if cal_data_file == 'camera.p':
            path = self.data_dir + '/' + cal_data_file
        else:
            path = cal_data_file

        if not os.path.exists(path):
            logger.debug('calibration data not found')
            return True

        with open(path, mode='rb') as f:
            camera_data = pickle.load(f)
            self.camera_mtx = camera_data['cameraMatrix']
            self.dist_coeffs = camera_data['distCoeffs']
            return False

        return True

    def undistort(self):
        """ Undistorts read images and writes them to files.

        Returns:
            Succeeded (False) or Failed (True)

        """
        logger.debug('undistort images used in camera calibration')

        if self.cal_img_names is None:
            self.read_cal_img_name()

        failed = False
        if self.camera_mtx is None or self.dist_coeffs is None:
            failed = self.load_calibration_data()
            if failed:
                failed = self.calibrate()

        if not failed:
            for name in self.cal_img_names:
                img = cv2.imread(name)
                undist = cv2.undistort(img,
                                       self.camera_mtx,
                                       self.dist_coeffs,
                                       None,
                                       self.camera_mtx)

                img_name = self.out_img_dir + '/' + os.path.basename(name)
                cv2.imwrite(img_name, undist)

        return failed

    def check_calibration(self, cal_data_file='camera.p'):
        """ Checks whether the camera calibration has done successfully.

        Returns:
            Succeeded (False) or Failed (True)

        """
        return self.has_calibrated() \
            or self.load_calibration_data(cal_data_file) \
            and self.calibrate()

    def has_calibrated(self):
        """ Checks whether the camera calibration has done or not.

        Returns:
            Not yet (False) or Done (True)

        """
        return self.camera_mtx and self.dist_coeffs

    def process_test_imgs(self, cal_data_file='camera.p', ext='jpg'):
        """ TODO: Add docstring
        """
        failed = self.check_calibration(cal_data_file)

        if not failed:
            path = self.test_img_dir + '/*' + ext
            for name in glob.glob(path):
                logger.debug('processing image: %s', name)
                img = mpimg.imread(name)
                warped, binary, undist = self.pre_process(img)

                ftitle, fext = os.path.splitext(os.path.basename(name))
                path = self.out_img_dir + '/' + ftitle + '_warped'+ fext
                mpimg.imsave(path, warped)
                path = self.out_img_dir + '/' + ftitle + '_binary'+ fext
                mpimg.imsave(path, binary)
                path = self.out_img_dir + '/' + ftitle + '_undist'+ fext
                mpimg.imsave(path, undist)

    def pre_process(self, img):
        """ TODO: Add docstring
        """
        undist = cv2.undistort(img,
                               self.camera_mtx,
                               self.dist_coeffs,
                               None,
                               self.camera_mtx)

        # Sobel x
        gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY).astype(np.float)
        sxbinary = abs_sobel_thresh(gray, thresh=(30, 255))

        # Threshold color channel
        hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS).astype(np.float)
        s_channel = hls[:, :, 2]
        s_binary = channel_thresh(s_channel, (170, 255))

        combined_binary = channel_or(sxbinary, s_binary)
        img_size = img.shape[0:2]
        img_size = img_size[::-1]

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

        mtx = cv2.getPerspectiveTransform(src, dst)
        self.invmtx = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(combined_binary, mtx, (img.shape[1], img.shape[0]))

        return warped, combined_binary, undist

class Line:
    """ Line data used by LineDetector
    """
    def __init__(self, xm_per_pix=1, ym_per_pix=1):
        """ TODO: Add docstring
        """
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []

        self.recent_x = []
        self.recent_y = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        self.best_fit_for_radius = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix

    def calc_current_x(self, y):
        """ TODO: Add docstring
        """
        return self.current_fit[0]*y**2 + self.current_fit[1]*y + self.current_fit[2]

    def calc_average_x(self, y):
        """ TODO: Add docstring
        """
        return self.best_fit[0]*y**2 + self.best_fit[1]*y + self.best_fit[2]

    def calc_radius(self, y, ym_per_pix=1):
        """ TODO: Add docstring
        """
        return ((1 + (2*self.best_fit_for_radius[0]*y*self.ym_per_pix
                      + self.best_fit_for_radius[1])**2)**1.5) \
                      / np.absolute(2*self.best_fit_for_radius[0])

    def update(self, current_x, current_y):
        if len(self.recent_x) > 5:
            self.recent_x.pop(0)
        self.recent_x.append(current_x)

        if len(self.recent_y) > 5:
            self.recent_y.pop(0)
        self.recent_y.append(current_y)

        all_x = [num for elem in self.recent_x for num in elem]
        all_y = [num for elem in self.recent_y for num in elem]

        self.best_fit = np.polyfit(all_y, all_x, 2)

        if self.xm_per_pix == 1 and self.ym_per_pix == 1:
            self.best_fit_for_radius = self.best_fit
        else:
            self.best_fit_for_radius \
                = np.polyfit(all_y*self.ym_per_pix, all_x*self.xm_per_pix, 2)

class LineDetector(ImageProcessor):
    """ A line detector
    """
    def __init__(self, data_dir, cal_img_dir, test_img_dir, out_img_dir):
        """ The initializer

        Args:
            data_dir: A directory used to load and save parameters.
            cal_img_dir: A directory where calibration images are read from.
            test_img_dir: A directory where test images are read from.
            out_img_dir: A directory where processed images are written.

        """
        super(LineDetector, self).__init__(data_dir, cal_img_dir, test_img_dir, out_img_dir)
        self.left_line = Line()
        self.right_line = Line()

        self.calibrated = False
        self.initial_search = True

    def find_line(self, img):
        """ TODO: Add docstring
        """
        if self.calibrated is False:
            self.check_calibration()
            self.calibrated = True

        binary_warped, _, undist = self.pre_process(img)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        nonzeroy, nonzerox = binary_warped.nonzero()
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        if self.initial_search:
            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :], axis=0)
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_current = np.argmax(histogram[:midpoint])
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(binary_warped.shape[0]/nwindows)

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Draw the windows on the visualization image
                '''
                cv2.rectangle(out_img,
                              (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high),
                              (0, 255, 0),
                              2)
                cv2.rectangle(out_img,
                              (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high),
                              (0, 255, 0),
                              2)
                '''
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low)
                                  & (nonzeroy < win_y_high)
                                  & (nonzerox >= win_xleft_low)
                                  & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low)
                                   & (nonzeroy < win_y_high)
                                   & (nonzerox >= win_xright_low)
                                   & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            self.initial_search = False

        else:
            left_lane_inds = ((nonzerox > (self.left_line.calc_average_x(nonzeroy) - margin))
                              & (nonzerox < (self.left_line.calc_average_x(nonzeroy) + margin)))
            right_lane_inds = ((nonzerox > (self.right_line.calc_average_x(nonzeroy) - margin))
                              & (nonzerox < (self.right_line.calc_average_x(nonzeroy) + margin)))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        self.left_line.update(leftx, lefty)
        self.right_line.update(rightx, righty)

        # Fit a second order polynomial to each
        #self.left_line.current_fit = np.polyfit(lefty, leftx, 2)
        #self.right_line.current_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = self.left_line.calc_average_x(ploty)
        right_fitx = self.right_line.calc_average_x(ploty)
        radius = self.left_line.calc_radius(np.max(ploty))

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.invmtx, (img.shape[1], img.shape[0]))

        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        '''
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(
            np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(
            np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))


        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        #cv2.putText(window_img, str(radius), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        '''
        '''
        plt.imshow(result)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        '''
        return result

def main():
    opts, _ = getopt.getopt(sys.argv[1:], 'fp', [])
    force = False
    plot = False
    for opt, _ in opts:
        if opt == '-f':
            force = True
        if opt == '-p':
            plot = True

    detector = LineDetector('../data',
                            '../camera_cal',
                            '../test_images',
                            '../output_images')
    if force:
        detector.calibrate()
        detector.undistort()

    if plot:
        detector.process_test_imgs()

    ## for a image
    #detector.find_line(mpimg.imread('../test_images/test3.jpg'))

    ## for a video
    clip = VideoFileClip('../project_video.mp4')
    output = '../output_images/project_video.mp4'

    #clip = VideoFileClip('../challenge_video.mp4')
    #output = '../output_images/challenge_video.mp4'

    #clip = VideoFileClip('../harder_challenge_video.mp4')
    #output = '../output_images/harder_challenge_video.mp4'

    clip.fl_image(detector.find_line).write_videofile(output, audio=False)

if __name__ == '__main__':
    main()
