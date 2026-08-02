import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.array(x, dtype=float)

    if rng is None:
        random_values = np.random.random(x.shape)
    else:
        random_values = rng.random(x.shape)

    mask = (random_values >= p).astype(float)
    scale = 1.0 / (1.0 - p)

    output = x * mask * scale
    dropout_pattern = mask * scale

    return output, dropout_pattern