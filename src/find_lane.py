#!/usr/bin/env python
"""Test

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

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(module)s: %(funcName)s: %(message)s'))
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

"""
image = mpimg.imread('test_images/straight_lines1.jpg')
plt.imshow(image)
plt.show()
"""

def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M

def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # 4) If corners found:
    if ret:
        # a) draw corners
        cv2.drawChessboardCorners(gray, (nx, ny), corners, ret)
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
             #Note: you could pick any four of the detected corners
             # as long as those four corners define a rectangle
             #One especially smart way to do this would be to use four well-chosen
             # corners that were automatically detected during the undistortion steps
             #We recommend using the automatic detection of corners in your code
        src = np.float32([
            corners[0].reshape(-1),
            corners[nx-1].reshape(-1),
            corners[-nx].reshape(-1),
            corners[-1].reshape(-1),
            ])
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        dst = np.float32([
            [100, 100],
            [img.shape[1]-100, 100],
            [100, img.shape[0]-100],
            [img.shape[1]-100, img.shape[0]-100]
            ])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        image_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(gray, M, image_size)
    else:
        M = None
        warped = np.copy(img)
    return warped, M

def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    """

    Args:
        img: An image to be processed
        orient:
        thresh:

    Returns:

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
    """Calculate gradient magnitude of an image and apply threshold

    Args:
        img: An image to be processed
        sobel_kernel:
        thresh:

    Returns:
        The processed binary image

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
    """A function to threshold an image for a given range and Sobel kernel

    Args:
        img: An image to be processed
        sobel_kernel:
        thresh:

    Returns:
        The processed binary image

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
    """Calculate gradient magnitude of an image and apply threshold

    Args:
        img: An image to be processed
        sobel_kernel:
        thresh:

    Returns:
        The processed binary image

    """
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(img)
    binary_output[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary_output

def channel_and(img1, img2):
    binary_output = np.zeros_like(img1)
    binary_output[(img1 == 1) & (img2 == 1)] = 1

    return binary_output

def channel_or(img1, img2):
    binary_output = np.zeros_like(img1)
    binary_output[(img1 == 1) | (img2 == 1)] = 1

    return binary_output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),
           max(0, int(center-width/2)):min(int(center+width/2), img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):

    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using
    # np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):, :int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):, int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum))-window_width/2+int(image.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height)
                                   :int(image.shape[0]-level*window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is
        # at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin, 0))
        l_max_index = int(min(l_center+offset+margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin, 0))
        r_max_index = int(min(r_center+offset+margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # 4) If corners found:
    if ret == True:
        # a) draw corners
        cv2.drawChessboardCorners(gray, (nx, ny), corners, ret)
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
             #Note: you could pick any four of the detected corners
             # as long as those four corners define a rectangle
             #One especially smart way to do this would be to use four well-chosen
             # corners that were automatically detected during the undistortion steps
             #We recommend using the automatic detection of corners in your code
        src = np.float32([
            corners[0].reshape(-1),
            corners[nx-1].reshape(-1),
            corners[-nx].reshape(-1),
            corners[-1].reshape(-1),
            ])
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        dst = np.float32([
            [100, 100],
            [img.shape[1]-100, 100],
            [100, img.shape[0]-100],
            [img.shape[1]-100, img.shape[0]-100]
            ])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        image_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(gray, M, image_size)
    else:
        M = None
        warped = np.copy(img)
    return warped, M

class ImageProcessor:
    def __init__(self, data_dir, cal_img_dir, test_img_dir, out_img_dir):
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

    def read_cal_img_name(self, ext='jpg'):
        path = self.cal_img_dir + '/*.' + ext
        self.cal_img_names = []
        for name in glob.glob(path):
            self.cal_img_names.append(name)

    def read_cal_img_shape(self):
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

    def calibrate(self, pattern_size=(9,6), cal_data_file='camera.p'):
        logger.debug('starts camera calibration')

        if self.cal_img_names is None:
            self.read_cal_img_name()

        if self.img_shape is None:
            self.read_cal_img_shape()

        nx, ny = pattern_size

        objpoint = np.zeros((np.prod((nx, ny)), 3), np.float32)
        objpoint[:, :2] = np.indices((nx, ny)).T.reshape(-1, 2)

        objpoints = []
        imgpoints = []
        for name in self.cal_img_names:
            img = cv2.imread(name)
            if self.img_shape != img.shape:
                logger.debug('detects different image size: %s', img.shape)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
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
        if 'camera.p' == cal_data_file:
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

    def process_test_imgs(self, cal_data_file='camera.p', ext='jpg'):
        failed = False
        if self.camera_mtx is None or self.dist_coeffs is None:
            failed = self.load_calibration_data(cal_data_file)
            if failed:
                failed = self.calibrate()

        if not failed:
            path = self.test_img_dir + '/*' + ext
            for name in glob.glob(path):
                logger.debug('processing image: %s', name)
                img = mpimg.imread(name)
                processed = self.pipeline(img)
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
                f.tight_layout()
                ax1.imshow(img)
                ax1.set_title('Original Image', fontsize=50)
                ax2.imshow(processed)
                ax2.set_title('Undistorted Image', fontsize=50)
                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
                plt.show()


    def pipeline(self, img):
        undist = cv2.undistort(img,
                               self.camera_mtx,
                               self.dist_coeffs,
                               None,
                               self.camera_mtx)

        hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS).astype(np.float)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x

        sxbinary = abs_sobel_thresh(s_channel, thresh=(20, 100))

        # Threshold color channel
        s_binary = channel_thresh(s_channel, (170, 255))

        dir_binary = dir_thresh(s_channel, thresh=(np.pi/4, np.pi/3))
        h_binary = channel_thresh(h_channel, (20,30))
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack((h_binary, channel_or(sxbinary, s_binary), np.zeros_like(s_binary)))
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
        warped = cv2.warpPerspective(color_binary, mtx, (img.shape[1], img.shape[0]))
        return warped


def main():
    opts, _ = getopt.getopt(sys.argv[1:], 'f', [])
    force = False
    for opt, _ in opts:
        if opt == '-f':
            force = True

    processor = ImageProcessor('../data', '../camera_cal', '../test_images', 'output_images')
    if force is True:
        processor.calibrate()
        processor.undistort()

    processor.process_test_imgs()
    '''
    camera_data_file = '../data/camera.p'
    if force or not os.path.exists('../data/camera.p'):
        logger.debug('starts camera calibration')

        _, mtx, dist, _, _ = calibrate('../camera_cal/',
                                       camera_data_file,
                                       (9, 6))

    else:
        logger.debug('loads camera calibration data')

        with open(camera_data_file, mode='rb') as f:
            camera_data = pickle.load(f)
            mtx = camera_data['cameraMatrix']
            dist = camera_data['distCoeffs']

    if force:
        logger.debug('undistorts calibration images')
        undistort('../camera_cal/', '../output_images', mtx, dist)

    logger.debug('run image pipeline')
    path = '../test_images/*.jpg'
    for name in glob.glob(path):
        print(name)
        img = mpimg.imread(name)
        processed = pipeline(img)
        out_name = '../output_images/' + os.path.basename(name)
        cv2.imwrite(out_name, processed)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(processed)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
'''
if __name__ == '__main__':
    main()
