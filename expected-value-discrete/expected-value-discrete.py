import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)

    if x.shape != p.shape:
        raise ValueError("Shapes must match")

    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1")

    return float(np.sum(x * p))