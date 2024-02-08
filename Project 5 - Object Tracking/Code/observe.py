import numpy as np
from chi2_cost import chi2_cost
from color_histogram import color_histogram


def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    num_particles = particles.shape[0]
    frame_height, frame_width, _ = frame.shape

    weight = np.zeros(num_particles, dtype=np.float64)
    for i in range(num_particles):

        x_min = np.rint(np.clip(particles[i, 0] - 0.5 * bbox_width, 0, frame_width-1)).astype(int)
        y_min = np.rint(np.clip(particles[i, 1] - 0.5 * bbox_height, 0, frame_height-1)).astype(int)
        x_max = np.rint(np.clip(particles[i, 0] + 0.5 * bbox_width, 0, frame_width-1)).astype(int)
        y_max = np.rint(np.clip(particles[i, 1] + 0.5 * bbox_height, 0, frame_height-1)).astype(int)

        # Calculate the color histogram (CH) for particle i
        hist_new = color_histogram(x_min, y_min, x_max, y_max, frame, hist_bin)

        # Calculate the distance between the CH of particle i and the current CH
        dist = chi2_cost(hist_new, hist)

        # Calculate the weight of particle i
        # Since we normalize the weights, we can use the guassian pdf without the normalization constant
        weight[i] = np.exp(-0.5 * (dist / sigma_observe)**2)
    
    # Normalize weights
    Z = np.sum(weight)
    if Z != 0: weight = weight / Z
    else: 
        weight = np.ones(num_particles) / num_particles
        print("Warning: All weights are zero. This might happen if the bounding box is too small or the model is too simple.")

    return np.expand_dims(weight, axis=1)