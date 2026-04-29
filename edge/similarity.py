"""Vector similarity operations for edge RAG."""

import math


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (list of floats)
        b: Second vector (list of floats)

    Returns:
        Cosine similarity score between -1.0 and 1.0.
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def rank_results(query_vec, chunks, top_k=5, threshold=0.0):
    """Rank chunks by cosine similarity to query vector.

    Args:
        query_vec: Query embedding vector
        chunks: List of dicts with "embedding" key
        top_k: Number of results to return
        threshold: Minimum similarity score (0.0 = no threshold)

    Returns:
        List of (chunk, score) tuples, sorted by score descending.
    """
    scored = []
    for chunk in chunks:
        if "embedding" not in chunk:
            continue
        score = cosine_similarity(query_vec, chunk["embedding"])
        if score >= threshold:
            scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
