import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    rows = len(A)
    cols = len(A[0])

    result = np.empty((cols, rows), dtype=float)

    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]

    return result
