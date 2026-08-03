def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended[:k] if item in relevant_set)

    precision = hits / k
    recall = hits / len(relevant_set)

    return [float(precision), float(recall)]