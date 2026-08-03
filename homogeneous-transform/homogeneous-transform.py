import numpy as np

def apply_homogeneous_transform(T, points):
    T = np.asarray(T)
    points = np.asarray(points)

    single_point = (points.ndim == 1)

    if single_point:
        points = points.reshape(1, 3)

    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.hstack((points, ones))

    # Apply transformation
    transformed_h = points_h @ T.T

    # Return only x, y, z
    result = transformed_h[:, :3]

    if single_point:
        return result[0]
    return result