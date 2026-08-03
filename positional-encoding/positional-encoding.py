import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    positions = np.arange(seq_len, dtype=float)[:, np.newaxis]

    div_term = np.power(
        base,
        (2 * np.arange((d_model + 1) // 2, dtype=float)) / d_model
    )

    angles = positions / div_term

    pe = np.empty((seq_len, d_model), dtype=float)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

    return pe