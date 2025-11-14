"""
Comprehensive Evaluation Script for LightGCN Model

Evaluates trained LightGCN on full test set with all metrics.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.graph_builder import build_graph
from src.models.lightgcn import create_lightgcn_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_recall(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Compute recall for single user."""
    if len(ground_truth) == 0:
        return 0.0
    hits = len(set(predictions) & set(ground_truth))
    return hits / len(ground_truth)


def compute_precision(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Compute precision for single user."""
    if len(predictions) == 0:
        return 0.0
    hits = len(set(predictions) & set(ground_truth))
    return hits / len(predictions)


def compute_ndcg(ground_truth: np.ndarray, predictions: np.ndarray, k: int) -> float:
    """Compute NDCG@K for single user."""
    relevance = np.array([1.0 if item in ground_truth else 0.0 for item in predictions[:k]])

    if relevance.sum() == 0:
        return 0.0

    # DCG
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

    # IDCG
    ideal_relevance = np.ones(min(len(ground_truth), k))
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def compute_hit_rate(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Compute hit rate."""
    return 1.0 if len(set(predictions) & set(ground_truth)) > 0 else 0.0


def evaluate_model(
    model,
    graph,
    test_interactions: Dict,
    user_to_idx: Dict,
    item_to_idx: Dict,
    k_values=[10, 20, 50],
    device="cuda",
    n_eval_users=None,
):
    """
    Comprehensive model evaluation.

    Args:
        model: Trained LightGCN model
        graph: User-item graph
        test_interactions: Dict mapping user_idx to list of item_idx
        user_to_idx: User ID to index mapping
        item_to_idx: Item ID to index mapping
        k_values: List of K values for top-K metrics
        device: Device to run on
        n_eval_users: Number of users to evaluate (None = all)

    Returns:
        Dictionary of metrics
    """
    model.eval()
    graph = graph.to(device)

    logger.info("=" * 80)
    logger.info("LightGCN Model Evaluation")
    logger.info("=" * 80)

    # Get all embeddings
    logger.info("Computing embeddings...")
    with torch.no_grad():
        user_all_emb, item_all_emb = model(graph)

    # Sample users if needed
    eval_users = list(test_interactions.keys())
    if n_eval_users and n_eval_users < len(eval_users):
        eval_users = np.random.choice(eval_users, size=n_eval_users, replace=False)

    logger.info(f"Evaluating on {len(eval_users)} users...")

    # Initialize metrics storage
    all_metrics = {k: {
        "recall": [],
        "precision": [],
        "ndcg": [],
        "hit_rate": []
    } for k in k_values}

    # Evaluate each user
    for user_idx in tqdm(eval_users, desc="Evaluating users"):
        # Get user embedding
        user_emb = user_all_emb[user_idx]

        # Compute scores for all items
        with torch.no_grad():
            scores = torch.matmul(item_all_emb, user_emb).cpu().numpy()

        # Get ground truth
        gt_items = np.array(test_interactions[user_idx])

        # Compute metrics for each K
        for k in k_values:
            # Get top-K items
            top_k_indices = np.argpartition(scores, -k)[-k:]
            top_k_items = top_k_indices[np.argsort(scores[top_k_indices])][::-1]

            # Compute metrics
            all_metrics[k]["recall"].append(compute_recall(gt_items, top_k_items))
            all_metrics[k]["precision"].append(compute_precision(gt_items, top_k_items))
            all_metrics[k]["ndcg"].append(compute_ndcg(gt_items, top_k_items, k))
            all_metrics[k]["hit_rate"].append(compute_hit_rate(gt_items, top_k_items))

    # Aggregate metrics
    final_metrics = {}
    for k in k_values:
        final_metrics[f"recall@{k}"] = np.mean(all_metrics[k]["recall"])
        final_metrics[f"precision@{k}"] = np.mean(all_metrics[k]["precision"])
        final_metrics[f"ndcg@{k}"] = np.mean(all_metrics[k]["ndcg"])
        final_metrics[f"hit_rate@{k}"] = np.mean(all_metrics[k]["hit_rate"])

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    for k in k_values:
        logger.info(f"\nTop-{k} Metrics:")
        logger.info(f"  Recall@{k}:    {final_metrics[f'recall@{k}']:.4f} ({final_metrics[f'recall@{k}']*100:.2f}%)")
        logger.info(f"  Precision@{k}: {final_metrics[f'precision@{k}']:.4f} ({final_metrics[f'precision@{k}']*100:.2f}%)")
        logger.info(f"  NDCG@{k}:      {final_metrics[f'ndcg@{k}']:.4f} ({final_metrics[f'ndcg@{k}']*100:.2f}%)")
        logger.info(f"  Hit Rate@{k}:  {final_metrics[f'hit_rate@{k}']:.4f} ({final_metrics[f'hit_rate@{k}']*100:.2f}%)")
    logger.info("=" * 80)

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LightGCN model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config_lightgcn.json",
        help="Path to training config",
    )
    parser.add_argument(
        "--n_users",
        type=int,
        default=None,
        help="Number of users to evaluate (None = all)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Build graph
    logger.info("Building graph...")
    data_dir = Path(config["data_dir"])

    graph_cache_file = Path(config.get("graph_cache_dir", "data/graph_cache")) / "lightgcn_graph.pt"
    if graph_cache_file.exists():
        logger.info("Loading cached graph...")
        cache = torch.load(graph_cache_file, weights_only=False)
        graph = cache["graph"]
        user_to_idx = cache["user_to_idx"]
        item_to_idx = cache["item_to_idx"]
    else:
        graph_obj = build_graph(
            n_users=config["n_users"],
            n_items=config["n_movies"],
            train_ratings_path=str(data_dir / "train_ratings.parquet"),
            min_rating=config["model"].get("rating_threshold", 4.0),
        )
        indices, values, size = graph_obj.get_sparse_graph()
        graph = torch.sparse.FloatTensor(indices, values, size)
        user_to_idx = graph_obj.user_to_idx
        item_to_idx = graph_obj.item_to_idx

    # Create model
    logger.info("Creating model...")
    model_config = {
        "n_users": config["n_users"],
        "n_movies": config["n_movies"],
        "embedding_dim": config["model"]["embedding_dim"],
        "n_layers": config["model"]["n_layers"],
        "dropout_rate": config["model"].get("dropout_rate", 0.0),
    }
    model = create_lightgcn_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    logger.info(f"Best metric during training: {checkpoint['best_metric']:.4f}")

    # Load test data
    logger.info("Loading test data...")
    test_df = pd.read_parquet(data_dir / "test_ratings.parquet")
    min_rating = config["model"].get("rating_threshold", 4.0)
    test_df = test_df[test_df["rating"] >= min_rating].copy()

    test_df["user_idx"] = test_df["userId"].map(user_to_idx)
    test_df["item_idx"] = test_df["movieId"].map(item_to_idx)
    test_df = test_df.dropna(subset=["user_idx", "item_idx"])
    test_df["user_idx"] = test_df["user_idx"].astype(int)
    test_df["item_idx"] = test_df["item_idx"].astype(int)

    test_interactions = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    logger.info(f"Test users: {len(test_interactions)}")

    # Evaluate
    metrics = evaluate_model(
        model=model,
        graph=graph,
        test_interactions=test_interactions,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        k_values=config["evaluation"]["k_values"],
        device=device,
        n_eval_users=args.n_users,
    )

    # Save results
    results_file = Path(args.checkpoint).parent / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
