import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage  # for the scipy.ndimage.maximum_filter
from scipy import signal  # for the scipy.signal.convolve2d function


# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    kernel_x = np.array([[0.0, 0.0, 0.0],
                         [-0.5, 0.0, 0.5],
                         [0.0, 0.0, 0.0]])
    gradient_x = signal.convolve2d(img, kernel_x, mode="same")
    
    kernel_y = np.array([[0.0, -0.5, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.5, 0.0]])
    gradient_y = signal.convolve2d(img, kernel_y, mode="same")

    
    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    # smoothed_x = cv2.GaussianBlur(gradient_x, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    # smoothed_y = cv2.GaussianBlur(gradient_y, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    smoothed_x = gradient_x
    smoothed_y = gradient_y

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    I_xx = ndimage.gaussian_filter(gradient_x**2, sigma)
    I_yy = ndimage.gaussian_filter(gradient_y**2, sigma)
    I_xy = ndimage.gaussian_filter(gradient_x * gradient_y, sigma)

    M = np.c_[I_xx.flatten(), 
              I_xy.flatten(),
              I_xy.flatten(),
              I_yy.flatten()].reshape(img.shape[0], img.shape[1], 2, 2)
    

    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    det_M = np.linalg.det(M)
    trace_M = np.trace(M, axis1=2, axis2=3)

    C = det_M - k * trace_M**2


    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format

    C_max_region = ndimage.maximum_filter(C, size=(3,3))
    corners = np.argwhere((C >= C_max_region) & (C > thresh))

    corners = np.c_[corners[:, 1], corners[:, 0]]

    return corners, C

