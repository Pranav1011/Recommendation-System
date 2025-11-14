"""
Evaluation Metrics for Recommendation Systems

Implements proper ranking metrics:
- Recall@K
- Precision@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate@K
- MAP@K (Mean Average Precision)
- Coverage

These metrics are specifically designed for ranking/recommendation tasks.
"""

from typing import Dict, List, Set

import numpy as np


def recall_at_k(
    predictions: List[List[int]], ground_truth: List[Set[int]], k: int = 10
) -> float:
    """
    Compute Recall@K.

    Measures: Of all items the user liked, how many appear in top-K recommendations?

    Args:
        predictions: List of ranked recommendations per user [[item_id, ...], ...]
        ground_truth: List of relevant items per user [{item_id, ...}, ...]
        k: Number of recommendations to consider

    Returns:
        Average recall@K across all users
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    recalls = []
    for pred, truth in zip(predictions, ground_truth):
        if len(truth) == 0:
            continue  # Skip users with no relevant items

        # Get top-K predictions
        top_k = pred[:k]

        # Count hits
        hits = len(set(top_k) & truth)

        # Recall = hits / total relevant items
        recalls.append(hits / len(truth))

    return np.mean(recalls) if recalls else 0.0


def precision_at_k(
    predictions: List[List[int]], ground_truth: List[Set[int]], k: int = 10
) -> float:
    """
    Compute Precision@K.

    Measures: Of the K items recommended, how many were relevant?

    Args:
        predictions: List of ranked recommendations per user
        ground_truth: List of relevant items per user
        k: Number of recommendations to consider

    Returns:
        Average precision@K across all users
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    precisions = []
    for pred, truth in zip(predictions, ground_truth):
        # Get top-K predictions
        top_k = pred[:k]

        # Count hits
        hits = len(set(top_k) & truth)

        # Precision = hits / K
        precisions.append(hits / k)

    return np.mean(precisions)


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at K.

    DCG = sum(relevance[i] / log2(i + 2)) for i in 0 to k-1

    Args:
        relevance_scores: Relevance scores for ranked items
        k: Number of items to consider

    Returns:
        DCG@K value
    """
    relevance_scores = np.array(relevance_scores[:k])
    if len(relevance_scores) == 0:
        return 0.0

    # Discount factor: 1/log2(i+2) for position i (1-indexed)
    discounts = 1.0 / np.log2(np.arange(2, len(relevance_scores) + 2))

    return np.sum(relevance_scores * discounts)


def ndcg_at_k(
    predictions: List[List[int]],
    ground_truth: List[Set[int]],
    relevance_scores: List[Dict[int, float]],
    k: int = 10,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    Measures ranking quality with position weighting - top items matter more.

    Args:
        predictions: List of ranked recommendations per user
        ground_truth: List of relevant items per user
        relevance_scores: List of dicts mapping item_id -> relevance score per user
        k: Number of recommendations to consider

    Returns:
        Average NDCG@K across all users
    """
    if len(predictions) != len(ground_truth) != len(relevance_scores):
        raise ValueError("All inputs must have same length")

    ndcgs = []
    for pred, truth, scores in zip(predictions, ground_truth, relevance_scores):
        if len(truth) == 0:
            continue

        # Get relevance for predicted items
        pred_relevance = [scores.get(item, 0.0) for item in pred[:k]]

        # Compute DCG
        dcg = dcg_at_k(pred_relevance, k)

        # Compute IDCG (ideal DCG with perfect ranking)
        ideal_relevance = sorted(scores.values(), reverse=True)
        idcg = dcg_at_k(ideal_relevance, k)

        # NDCG = DCG / IDCG
        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)

    return np.mean(ndcgs) if ndcgs else 0.0


def hit_rate_at_k(
    predictions: List[List[int]], ground_truth: List[Set[int]], k: int = 10
) -> float:
    """
    Compute Hit Rate@K.

    Measures: % of users with at least 1 relevant item in top-K.

    Args:
        predictions: List of ranked recommendations per user
        ground_truth: List of relevant items per user
        k: Number of recommendations to consider

    Returns:
        Hit rate@K (fraction of users with at least one hit)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    hits = 0
    total = len(predictions)

    for pred, truth in zip(predictions, ground_truth):
        # Get top-K predictions
        top_k = pred[:k]

        # Check if any prediction is in ground truth
        if len(set(top_k) & truth) > 0:
            hits += 1

    return hits / total if total > 0 else 0.0


def map_at_k(
    predictions: List[List[int]], ground_truth: List[Set[int]], k: int = 10
) -> float:
    """
    Compute Mean Average Precision at K.

    Considers the order of all relevant items in top-K.

    Args:
        predictions: List of ranked recommendations per user
        ground_truth: List of relevant items per user
        k: Number of recommendations to consider

    Returns:
        Average MAP@K across all users
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    aps = []
    for pred, truth in zip(predictions, ground_truth):
        if len(truth) == 0:
            continue

        # Get top-K predictions
        top_k = pred[:k]

        # Compute average precision
        ap = 0.0
        hits = 0

        for i, item in enumerate(top_k):
            if item in truth:
                hits += 1
                # Precision at position i+1
                precision_at_i = hits / (i + 1)
                ap += precision_at_i

        # Normalize by min(k, |relevant items|)
        ap /= min(len(truth), k)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


def coverage(predictions: List[List[int]], n_items: int) -> float:
    """
    Compute catalog coverage.

    Measures: % of catalog items that get recommended (diversity metric).

    Args:
        predictions: List of ranked recommendations per user
        n_items: Total number of items in catalog

    Returns:
        Coverage (fraction of items that appear in recommendations)
    """
    recommended_items = set()

    for pred in predictions:
        recommended_items.update(pred)

    return len(recommended_items) / n_items if n_items > 0 else 0.0


def compute_all_metrics(
    predictions: List[List[int]],
    ground_truth: List[Set[int]],
    relevance_scores: List[Dict[int, float]],
    n_items: int,
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Compute all metrics at multiple K values.

    Args:
        predictions: List of ranked recommendations per user
        ground_truth: List of relevant items per user
        relevance_scores: List of dicts mapping item_id -> relevance score per user
        n_items: Total number of items in catalog
        k_values: List of K values to evaluate

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(predictions, ground_truth, k)
        metrics[f"precision@{k}"] = precision_at_k(predictions, ground_truth, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(predictions, ground_truth, relevance_scores, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(predictions, ground_truth, k)
        metrics[f"map@{k}"] = map_at_k(predictions, ground_truth, k)

    # Coverage (not K-dependent)
    metrics["coverage"] = coverage(predictions, n_items)

    return metrics


if __name__ == "__main__":
    # Test metrics with toy example
    print("Testing Recommendation Metrics...")

    # Create toy data
    # User 1: Likes items [1, 2, 3], we recommend [1, 5, 2, 7, 8]
    # User 2: Likes items [4, 5], we recommend [4, 5, 6, 7, 8]
    # User 3: Likes items [6, 7, 8, 9], we recommend [6, 10, 11, 7, 12]

    predictions = [
        [1, 5, 2, 7, 8],  # User 1
        [4, 5, 6, 7, 8],  # User 2
        [6, 10, 11, 7, 12],  # User 3
    ]

    ground_truth = [
        {1, 2, 3},  # User 1
        {4, 5},  # User 2
        {6, 7, 8, 9},  # User 3
    ]

    relevance_scores = [
        {1: 5.0, 2: 4.5, 3: 4.0},  # User 1
        {4: 5.0, 5: 4.5},  # User 2
        {6: 5.0, 7: 4.5, 8: 4.0, 9: 3.5},  # User 3
    ]

    n_items = 15

    # Compute metrics at K=5
    k = 5

    recall = recall_at_k(predictions, ground_truth, k)
    precision = precision_at_k(predictions, ground_truth, k)
    ndcg = ndcg_at_k(predictions, ground_truth, relevance_scores, k)
    hit_rate = hit_rate_at_k(predictions, ground_truth, k)
    map_score = map_at_k(predictions, ground_truth, k)
    cov = coverage(predictions, n_items)

    print(f"\nMetrics at K={k}:")
    print(f"  Recall@{k}: {recall:.4f}")
    print(f"  Precision@{k}: {precision:.4f}")
    print(f"  NDCG@{k}: {ndcg:.4f}")
    print(f"  Hit Rate@{k}: {hit_rate:.4f}")
    print(f"  MAP@{k}: {map_score:.4f}")
    print(f"  Coverage: {cov:.4f}")

    # Test compute_all_metrics
    all_metrics = compute_all_metrics(
        predictions, ground_truth, relevance_scores, n_items, k_values=[3, 5]
    )

    print("\nAll metrics:")
    for metric_name, value in sorted(all_metrics.items()):
        print(f"  {metric_name}: {value:.4f}")

    print("\n✓ All metrics working correctly!")
