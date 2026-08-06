import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum(y_true == y_pred)
    fp = np.sum(y_true != y_pred)
    fn = fp

    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float((2 * tp) / denom)