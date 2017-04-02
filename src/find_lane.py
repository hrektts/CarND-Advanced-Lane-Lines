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
    """ Apply Sobel operator and threshold to an image.

    Args:
        img: A single channel image to be processed
        orient: 'x' or 'y': The direction used to calculate the derivative
        thresh: Minimum and maximum thresholds used to binalize the processed image

    Returns:
        A processed binary image

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
    """ Calculate gradient magnitude of an image and apply threshold.

    Args:
        img: A single channel image to be processed
        sobel_kernel: The kernel size used by Sobel operator
        thresh: Minimum and maximum thresholds used to binalize the processed image

    Returns:
        A processed binary image

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
    """ Threshold an image for a given range and Sobel kernel.

    Args:
        img: A single channel image to be processed
        sobel_kernel: The kernel size used by Sobel operator
        thresh: Minimum and maximum thresholds used to binalize the processed image

    Returns:
        A processed binary image

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
    """ Calculate gradient magnitude of an image and applies threshold.

    Args:
        img: A single channel image to be processed
        thresh: Minimum and maximum thresholds used to binalize the processed image

    Returns:
        A processed binary image

    """
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(img)
    binary_output[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary_output

class ImageProcessor:
    """ A image processor
    """
    def __init__(self, data_dir, cal_img_dir, test_img_dir, out_img_dir):
        """ The initializer

        Args:
            data_dir: A directory used to load and save parameters
            cal_img_dir: A directory where calibration images are read from
            test_img_dir: A directory where test images are read from
            out_img_dir: A directory where processed images are written

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
        """ Read all image names in the calibration image directory.

        Args:
            ext: The extention of the images

        """
        path = self.cal_img_dir + '/*.' + ext
        self.cal_img_names = []
        for name in glob.glob(path):
            self.cal_img_names.append(name)

    def read_cal_img_shape(self):
        """ Read the shape of images.
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
        """ Calibrate camera using chessboard images.

        Args:
            pattern_size: The number of crossing points of chessboard image
                          (x-direction, y-direction)
            cal_data_file: A file name used to store calibration data

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

        succeeded, mtx, dist, _, _ = cv2.calibrateCamera(objpoints,
                                                         imgpoints,
                                                         self.img_shape[0:2],
                                                         None,
                                                         None)
        if succeeded:
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
        """ Load camera calibration data.

        Args:
            cal_data_file: A file name used to read calibration data

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
        """ Undistort read images and writes them to files.

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
        """ Check whether the camera calibration has done successfully.

        Returns:
            Succeeded (False) or Failed (True)

        """
        return not self.has_calibrated() \
            and self.load_calibration_data(cal_data_file) \
            and self.calibrate()

    def has_calibrated(self):
        """ Check whether the camera calibration has done or not.

        Returns:
            Not yet (False) or Done (True)

        """
        return self.camera_mtx is not None and self.dist_coeffs is not None

    def process_test_imgs(self, cal_data_file='camera.p', ext='jpg'):
        """ TODO: Add docstring
        """
        failed = self.check_calibration(cal_data_file)

        if not failed:
            path = self.test_img_dir + '/*' + ext
            for name in glob.glob(path):
                logger.debug('processing image: %s', name)
                img = mpimg.imread(name)
                warped, binary, undist = self.do_pre_processing(img)

                ftitle, fext = os.path.splitext(os.path.basename(name))

                path = self.out_img_dir + '/' + ftitle + '_warped'+ fext
                out = np.dstack((warped, warped, warped))
                mpimg.imsave(path, out)
                path = self.out_img_dir + '/' + ftitle + '_binary'+ fext
                out = np.dstack((binary, binary, binary))
                mpimg.imsave(path, out)
                path = self.out_img_dir + '/' + ftitle + '_undist'+ fext
                mpimg.imsave(path, undist)

    def do_pre_processing(self, img):
        """ TODO: Add docstring
        """
        undist = cv2.undistort(img,
                               self.camera_mtx,
                               self.dist_coeffs,
                               None,
                               self.camera_mtx)

        # Sobel x
        gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY).astype(np.float)
        sxbinary = abs_sobel_thresh(gray, thresh=(30, 255)).astype(np.float)

        # Threshold color channel
        hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS).astype(np.float)
        s_channel = hls[:, :, 2]
        s_binary = channel_thresh(s_channel, (100, 255)).astype(np.float)

        h_channel = hls[:, :, 0]
        yellow_binary = channel_thresh(h_channel, (15, 35)).astype(np.float)

        combined_binary = cv2.bitwise_and(sxbinary, yellow_binary)
        combined_binary = cv2.bitwise_or(combined_binary, s_binary)
        combined_binary = cv2.bitwise_or(combined_binary, sxbinary)

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
    """ Line data
    """
    def __init__(self, xm_per_pix=1, ym_per_pix=1):
        """ TODO: Add docstring
        """
        # x values of the last n detection
        self.recent_x = []
        # y values of the last n detection
        self.recent_y = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit_for_radius = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # meters per pixel in x dimension
        self.xm_per_pix = xm_per_pix
        # meters per pixel in y dimension
        self.ym_per_pix = ym_per_pix

    def calc_average_x(self, y):
        """ TODO: Add docstring
        """
        return self.best_fit[0]*y**2 + self.best_fit[1]*y + self.best_fit[2]

    def calc_radius(self, y):
        """ TODO: Add docstring
        """
        self.radius_of_curvature = ((1 + (2*self.best_fit_for_radius[0]*y*self.ym_per_pix
                                          + self.best_fit_for_radius[1])**2)**1.5) \
                                          / np.absolute(2*self.best_fit_for_radius[0])

    def update(self, current_x, current_y):
        """ TODO: Add docstring
        """
        if len(self.recent_x) > 5:
            self.recent_x.pop(0)
        self.recent_x.append(current_x)

        if len(self.recent_y) > 5:
            self.recent_y.pop(0)
        self.recent_y.append(current_y)

        all_x = np.array([num for elem in self.recent_x for num in elem]).astype(np.float)
        all_y = np.array([num for elem in self.recent_y for num in elem]).astype(np.float)

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
        self.xm_per_pix = 3.7/700
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.left_line = Line(self.xm_per_pix, self.ym_per_pix)
        self.right_line = Line(self.xm_per_pix, self.ym_per_pix)

        self.calibrated = False
        self.initial_search = True

    def process_test_imgs(self, cal_data_file='camera.p', ext='jpg'):
        """ TODO: Add docstring
        """
        super(LineDetector, self).process_test_imgs(cal_data_file, ext)

        failed = self.check_calibration(cal_data_file)

        if not failed:
            path = self.test_img_dir + '/*' + ext
            for name in glob.glob(path):
                logger.debug('processing image: %s', name)

                img = mpimg.imread(name)
                self.initial_search = True
                for i in range(10):
                    result, lined = self.find_line(img, test=True)

                ftitle, fext = os.path.splitext(os.path.basename(name))

                path = self.out_img_dir + '/' + ftitle + '_result'+ fext
                mpimg.imsave(path, result)
                path = self.out_img_dir + '/' + ftitle + '_lined'+ fext
                mpimg.imsave(path, lined)

    def find_line(self, img, test=False):
        """ TODO: Add docstring
        """
        if self.calibrated is False:
            self.check_calibration()
            self.calibrated = True

        binary_warped, _, undist = self.do_pre_processing(img)

        nonzeroy, nonzerox = binary_warped.nonzero()
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        if self.initial_search:
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :], axis=0)
            #plt.plot(histogram)
            #plt.xlabel('Pixel Position', fontsize=12)
            #plt.ylabel('Counts', fontsize=12)
            #plt.show()

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_current = np.argmax(histogram[:midpoint])
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 12
            # Set height of windows
            window_height = np.int(binary_warped.shape[0]/nwindows)

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            #out_img = out_img = np.dstack((binary_warped, binary_warped, binary_warped)).astype(np.uint8)*255

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

            #plt.imshow(out_img)
            #plt.show()

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

        # Fit a second order polynomial to each
        self.left_line.update(leftx, lefty)
        self.right_line.update(rightx, righty)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = self.left_line.calc_average_x(ploty)
        right_fitx = self.right_line.calc_average_x(ploty)

        # Calculate corner radius at the bottom of the image
        self.left_line.calc_radius(np.max(ploty))
        self.right_line.calc_radius(np.max(ploty))

        if test:
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)).astype(np.uint8)

            window_img = np.zeros_like(out_img).astype(np.uint8)
            line_img = np.zeros_like(out_img).astype(np.uint8)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            cv2.polylines(line_img, np.int32(pts_left), False, (255, 255, 0), 2)
            cv2.polylines(line_img, np.int32(pts_right), False, (255, 255, 0), 2)
            lined = cv2.addWeighted(out_img, 1, line_img, 1, 0)
            lined = cv2.addWeighted(lined, 1, window_img, 0.3, 0)

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix
        newwarp = cv2.warpPerspective(color_warp, self.invmtx, (img.shape[1], img.shape[0]))

        # Calculate vhecle offset
        vehicle_offset = np.mean([right_fitx[-1], left_fitx[-1]]) \
                         - binary_warped.shape[1]/2
        vehicle_offset = vehicle_offset * self.xm_per_pix
        if vehicle_offset < 0:
            offset_direction = 'right'
            vehicle_offset = np.absolute(vehicle_offset)
        else:
            offset_direction = 'left'

        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            result,
            'Radius of the left line: {0:9.2f} m'.format(self.left_line.radius_of_curvature),
            (30, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            result,
            'Radius of the right line: {0:8.2f} m'.format(self.right_line.radius_of_curvature),
            (30, 80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            result,
            'Vehicle is {0:.2f} m {1} of center'.format(vehicle_offset, offset_direction),
            (30, 110), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if test:
            return result, lined
        else:
            return result


def main():
    opts, _ = getopt.getopt(sys.argv[1:], 'ft', [])
    force = False
    test = False
    for opt, _ in opts:
        if opt == '-f':
            force = True
        if opt == '-t':
            test = True

    detector = LineDetector('../data',
                            '../camera_cal',
                            '../test_images',
                            '../output_images')
    if force:
        detector.calibrate()
        detector.undistort()

    if test:
        detector.process_test_imgs()
    else:
        ## for video processing
        clip = VideoFileClip('../project_video.mp4')
        output = '../output_images/project_video.mp4'

        #clip = VideoFileClip('../challenge_video.mp4')
        #output = '../output_images/challenge_video.mp4'

        #clip = VideoFileClip('../harder_challenge_video.mp4')
        #output = '../output_images/harder_challenge_video.mp4'

        clip.fl_image(detector.find_line).write_videofile(output, audio=False)

if __name__ == '__main__':
    main()
