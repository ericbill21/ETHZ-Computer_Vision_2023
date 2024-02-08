import numpy as np
from scipy.spatial.distance import cdist


def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please

    def ssd_metric(A, B):
        return np.sum((A - B)**2)
    result = cdist(desc1, desc2, metric=ssd_metric)
    
    return result

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        
        matches = np.c_[np.arange(q1), np.argmin(distances, axis=1)]

    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis

        # Find the nearest neighbor for each descriptor of each imgage
        matches1 = np.c_[np.arange(q1), np.argmin(distances, axis=1)].tolist()
        matches2 = np.c_[np.argmin(distances, axis=0), np.arange(q2)].tolist()

        # Calculate the intersection of matches1 and matches2
        matches = np.array([m for m in matches1 if m in matches2])
        
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row

        first_nn = np.argmin(distances, axis=1)
        first_nn_distances = np.min(distances, axis=1)
        matches = np.c_[np.arange(q1), first_nn]

        distances[matches[:, 0], matches[:, 1]] = np.inf

        second_nn = np.argmin(distances, axis=1)
        second_nn_distances = np.min(distances, axis=1)

        ratio = first_nn_distances / second_nn_distances
        index = np.argwhere(ratio < ratio_thresh).flatten()

        matches = np.c_[np.arange(q1), first_nn][index]

    else:
        raise NotImplementedError
    return matches

