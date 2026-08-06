import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    H = np.asarray(params["bz"]).shape[0]
    D = np.asarray(params["Wz"]).shape[0]

    x, x_1d = _as2d(x, D)
    h_prev, h_1d = _as2d(h_prev, H)

    Wz = np.asarray(params["Wz"], dtype=float)
    Uz = np.asarray(params["Uz"], dtype=float)
    bz = np.asarray(params["bz"], dtype=float)

    Wr = np.asarray(params["Wr"], dtype=float)
    Ur = np.asarray(params["Ur"], dtype=float)
    br = np.asarray(params["br"], dtype=float)

    Wh = np.asarray(params["Wh"], dtype=float)
    Uh = np.asarray(params["Uh"], dtype=float)
    bh = np.asarray(params["bh"], dtype=float)

    z = _sigmoid(x @ Wz + h_prev @ Uz + bz)
    r = _sigmoid(x @ Wr + h_prev @ Ur + br)
    h_tilde = np.tanh(x @ Wh + (r * h_prev) @ Uh + bh)

    h = (1.0 - z) * h_prev + z * h_tilde

    return h[0] if (x_1d and h_1d) else h