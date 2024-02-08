import numpy as np


def resample(particles, particles_w):
    num_particles = particles.shape[0]
    
    # Resample N indices according to their probabilities
    samples = np.random.choice(np.arange(num_particles),
                            size=num_particles,
                            replace=True,
                            p=particles_w[:, 0])
    
    # Reassign particles according to samples and normalize weights
    particles, particles_w = particles[samples], particles_w[samples]
    particles_w /= np.sum(particles_w[:, 0])
    
    return (particles, particles_w) 