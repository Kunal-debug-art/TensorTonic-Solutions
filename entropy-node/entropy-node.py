import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)

    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)

    probabilities = counts / len(y)

    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)