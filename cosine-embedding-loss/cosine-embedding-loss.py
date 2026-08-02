def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code heredef cosine_embedding_loss(x1, x2, label, margin):
    dot = sum(a * b for a, b in zip(x1, x2))
    norm1 = sum(a * a for a in x1) ** 0.5
    norm2 = sum(b * b for b in x2) ** 0.5

    cosine_similarity = dot / (norm1 * norm2)

    if label == 1:
        return float(1 - cosine_similarity)
    else:
        return float(max(0.0, cosine_similarity - margin))