import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """
    Compute reset gate: r_t = sigmoid(W_r @ [h, x] + b_r)
    """
    h_prev = np.asarray(h_prev, dtype=float)
    x_t = np.asarray(x_t, dtype=float)
    W_r = np.asarray(W_r, dtype=float)
    b_r = np.asarray(b_r, dtype=float)

    single = h_prev.ndim == 1

    if single:
        h_prev = h_prev.reshape(1, -1)
        x_t = x_t.reshape(1, -1)

    concat = np.concatenate([h_prev, x_t], axis=1)

    result = sigmoid(concat @ W_r.T + b_r)

    return result[0] if single else result