import time

import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import color, io
from skimage.transform import rescale


def distance(x, X):
    return np.linalg.norm(X - x, axis=1)

# return 1 / (bandwidth * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (dist / bandwidth)**2)
#TODO: CHECK LOGIC: we dont need the scalar as it is a constant
def gaussian(dist, bandwidth):
    return np.exp(-0.5 * (dist / bandwidth)**2)

def update_point(weight, X):
    numerator = np.sum(X * weight[:, np.newaxis], axis=0)
    denominator = np.sum(weight)
    new_point = numerator / denominator
    return new_point

def meanshift_step(X, bandwidth=1.0):
    result = np.zeros_like(X)
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        result[i] = update_point(weight, X)

    return result

def meanshift(X):
    for _ in range(20):
        X = meanshift_step(X)
        print(f'Iteration {_ + 1} done', end='\r')
    return X

scale = 0.5 # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(image_lab)
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

# If there are more centroids than colors, use the centroids as the result
if len(centroids) > len(colors):
    # Normalize centroids to [0, 1] range
    centroids = (centroids - centroids.min()) / (centroids.max() - centroids.min())
    result_image = centroids[labels].reshape(shape)     
else:
    result_image = colors[labels].reshape(shape)

result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
