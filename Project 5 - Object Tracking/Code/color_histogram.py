import numpy as np


def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    box = frame[ymin:ymax, xmin:xmax]

    R = np.histogram(box[:, :, 0], bins=hist_bin, range=(0, 255))[0]
    G = np.histogram(box[:, :, 1], bins=hist_bin, range=(0, 255))[0]
    B = np.histogram(box[:, :, 2], bins=hist_bin, range=(0, 255))[0]

    return np.concatenate((R, G, B)) / np.sum(R + G + B)