import numpy as np


def estimate(particles, particles_w):
    return np.average(particles, axis=0, weights=particles_w[:, 0])