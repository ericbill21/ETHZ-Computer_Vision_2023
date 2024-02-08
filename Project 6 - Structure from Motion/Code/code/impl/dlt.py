import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  for i in range(num_corrs):
    constraint_rows = np.zeros((2, 12))

    # Using homogeneous coordinates for the 3D points here makes the code simpler
    X = np.append(points3D[i,:], 1)
    x = points2D[i,:]

    constraint_rows[0, 0:4] = X
    constraint_rows[0, 8:12] = -x[0] * X
    constraint_rows[1, 4:8] = -X
    constraint_rows[1, 8:12] = x[1] * X

    constraint_matrix[2*i:2*(i+1), :] = constraint_rows

  return constraint_matrix