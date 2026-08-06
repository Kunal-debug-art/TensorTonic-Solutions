import numpy as np

def euclidean_distance(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.sum((x - y) ** 2)))