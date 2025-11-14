"""
Evaluate Two-Tower Model Performance

Computes ranking metrics on test set:
- Recall@K
- Precision@K
- NDCG@K
- Hit Rate@K
- MAP@K
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_embeddings(embeddings_dir: Path) -> tuple:
    """Load user and movie embeddings."""
    logger.info(f"Loading embeddings from {embeddings_dir}")

    user_embeddings = np.load(embeddings_dir / "user_embeddings.npy")
    movie_embeddings = np.load(embeddings_dir / "movie_embeddings.npy")

    logger.info(f"User embeddings: {user_embeddings.shape}")
    logger.info(f"Movie embeddings: {movie_embeddings.shape}")

    return user_embeddings, movie_embeddings


def load_test_data(test_path: Path) -> pd.DataFrame:
    """Load test ratings."""
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_parquet(test_path)
    logger.info(f"Loaded {len(test_df):,} test ratings")
    return test_df


def create_user_movie_mappings(test_df: pd.DataFrame, train_df: pd.DataFrame = None):
    """
    Create user/movie ID to index mappings.

    Returns mappings that align with the dataset's internal indexing.
    """
    # Get all unique users and movies (sorted)
    if train_df is not None:
        all_users = sorted(
            set(test_df["userId"].unique()) | set(train_df["userId"].unique())
        )
        all_movies = sorted(
            set(test_df["movieId"].unique()) | set(train_df["movieId"].unique())
        )
    else:
        all_users = sorted(test_df["userId"].unique())
        all_movies = sorted(test_df["movieId"].unique())

    user_to_idx = {user_id: idx for idx, user_id in enumerate(all_users)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(all_movies)}

    return user_to_idx, movie_to_idx


def get_top_k_recommendations(
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
    user_indices: List[int],
    k: int = 100,
    batch_size: int = 1000,
) -> List[List[int]]:
    """
    Get top-K movie recommendations for each user.

    Args:
        user_embeddings: User embedding matrix (n_users, embed_dim)
        movie_embeddings: Movie embedding matrix (n_movies, embed_dim)
        user_indices: List of user indices to generate recommendations for
        k: Number of recommendations per user
        batch_size: Batch size for processing

    Returns:
        List of top-K movie indices for each user
    """
    recommendations = []

    # Convert to torch for faster computation
    movie_emb_tensor = torch.tensor(movie_embeddings, dtype=torch.float32)

    logger.info(f"Generating top-{k} recommendations for {len(user_indices)} users...")

    for i in tqdm(range(0, len(user_indices), batch_size), desc="Batches"):
        batch_user_indices = user_indices[i : i + batch_size]
        batch_user_emb = user_embeddings[batch_user_indices]

        # Convert to tensor
        user_emb_tensor = torch.tensor(batch_user_emb, dtype=torch.float32)

        # Compute similarity: (batch_size, n_movies)
        similarities = torch.mm(user_emb_tensor, movie_emb_tensor.T)

        # Get top-K for each user
        _, top_k_indices = torch.topk(similarities, k, dim=1)

        recommendations.extend(top_k_indices.cpu().numpy().tolist())

    return recommendations


def prepare_ground_truth(
    test_df: pd.DataFrame,
    user_to_idx: Dict[int, int],
    movie_to_idx: Dict[int, int],
    rating_threshold: float = 4.0,
) -> tuple:
    """
    Prepare ground truth for evaluation.

    Args:
        test_df: Test dataframe
        user_to_idx: User ID to index mapping
        movie_to_idx: Movie ID to index mapping
        rating_threshold: Minimum rating to consider as relevant

    Returns:
        Tuple of (user_indices, ground_truth_sets, relevance_scores)
    """
    # Filter test set for users/movies we have embeddings for
    test_df = test_df[
        test_df["userId"].isin(user_to_idx.keys())
        & test_df["movieId"].isin(movie_to_idx.keys())
    ].copy()

    # Map IDs to indices
    test_df["user_idx"] = test_df["userId"].map(user_to_idx)
    test_df["movie_idx"] = test_df["movieId"].map(movie_to_idx)

    # Group by user
    user_groups = test_df.groupby("user_idx")

    user_indices = []
    ground_truth = []
    relevance_scores = []

    for user_idx, group in user_groups:
        user_indices.append(user_idx)

        # Ground truth: movies with rating >= threshold
        relevant_movies = set(
            group[group["rating"] >= rating_threshold]["movie_idx"].tolist()
        )
        ground_truth.append(relevant_movies)

        # Relevance scores: use actual ratings
        scores = dict(zip(group["movie_idx"], group["rating"]))
        relevance_scores.append(scores)

    logger.info(f"Prepared ground truth for {len(user_indices)} users")
    logger.info(
        f"Avg relevant items per user: {np.mean([len(gt) for gt in ground_truth]):.1f}"
    )

    return user_indices, ground_truth, relevance_scores


def compute_metrics(
    predictions: List[List[int]],
    ground_truth: List[Set[int]],
    relevance_scores: List[Dict[int, float]],
    n_items: int,
    k_values: List[int] = [5, 10, 20, 50],
) -> Dict[str, float]:
    """Compute all ranking metrics."""
    from src.training.metrics import (
        coverage,
        hit_rate_at_k,
        map_at_k,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
    )

    metrics = {}

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(predictions, ground_truth, k)
        metrics[f"precision@{k}"] = precision_at_k(predictions, ground_truth, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(predictions, ground_truth, relevance_scores, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(predictions, ground_truth, k)
        metrics[f"map@{k}"] = map_at_k(predictions, ground_truth, k)

    metrics["coverage"] = coverage(predictions, n_items)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Two-Tower model")
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="data/embeddings",
        help="Directory containing embeddings",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test_ratings.parquet",
        help="Path to test ratings",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train_ratings.parquet",
        help="Path to train ratings (for mapping)",
    )
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=4.0,
        help="Minimum rating to consider as relevant",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of recommendations to generate per user",
    )

    args = parser.parse_args()

    # Load data
    embeddings_dir = Path(args.embeddings_dir)
    user_embeddings, movie_embeddings = load_embeddings(embeddings_dir)

    test_df = load_test_data(Path(args.test_data))
    train_df = pd.read_parquet(Path(args.train_data))

    # Create mappings
    user_to_idx, movie_to_idx = create_user_movie_mappings(test_df, train_df)

    logger.info(f"Total users: {len(user_to_idx)}")
    logger.info(f"Total movies: {len(movie_to_idx)}")

    # Prepare ground truth
    user_indices, ground_truth, relevance_scores = prepare_ground_truth(
        test_df, user_to_idx, movie_to_idx, args.rating_threshold
    )

    # Generate recommendations
    predictions = get_top_k_recommendations(
        user_embeddings, movie_embeddings, user_indices, k=args.top_k
    )

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(
        predictions,
        ground_truth,
        relevance_scores,
        n_items=len(movie_to_idx),
        k_values=[5, 10, 20, 50],
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Rating threshold: {args.rating_threshold}")
    logger.info(f"Number of test users: {len(user_indices)}")
    logger.info(f"Number of movies: {len(movie_to_idx)}")
    logger.info("")

    for metric_name in sorted(metrics.keys()):
        value = metrics[metric_name]
        logger.info(f"{metric_name:20s}: {value:.4f}")

    logger.info("=" * 60)

    # Interpretation
    logger.info("\n" + "=" * 60)
    logger.info("INTERPRETATION")
    logger.info("=" * 60)

    recall_10 = metrics.get("recall@10", 0)
    ndcg_10 = metrics.get("ndcg@10", 0)

    if recall_10 >= 0.20:
        logger.info("✅ EXCELLENT: Recall@10 >= 0.20 (industry standard)")
    elif recall_10 >= 0.15:
        logger.info("✓ GOOD: Recall@10 >= 0.15 (acceptable)")
    elif recall_10 >= 0.10:
        logger.info("⚠ FAIR: Recall@10 >= 0.10 (needs improvement)")
    else:
        logger.info("❌ POOR: Recall@10 < 0.10 (retrain recommended)")

    if ndcg_10 >= 0.25:
        logger.info("✅ EXCELLENT: NDCG@10 >= 0.25 (strong ranking)")
    elif ndcg_10 >= 0.20:
        logger.info("✓ GOOD: NDCG@10 >= 0.20 (decent ranking)")
    elif ndcg_10 >= 0.15:
        logger.info("⚠ FAIR: NDCG@10 >= 0.15 (weak ranking)")
    else:
        logger.info("❌ POOR: NDCG@10 < 0.15 (retrain recommended)")

    logger.info("=" * 60)

    # Recommendations
    if recall_10 < 0.15 or ndcg_10 < 0.20:
        logger.info("\n" + "=" * 60)
        logger.info("RECOMMENDATIONS FOR IMPROVEMENT")
        logger.info("=" * 60)
        logger.info("Consider retraining with:")
        logger.info("  1. BPR loss instead of MSE (better for ranking)")
        logger.info("  2. Negative sampling (n_negatives=4)")
        logger.info("  3. Larger embedding dimension (256 instead of 128)")
        logger.info("  4. More training epochs (current early stopped at epoch 6)")
        logger.info("  5. Lower learning rate (0.0001 instead of 0.0005)")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
