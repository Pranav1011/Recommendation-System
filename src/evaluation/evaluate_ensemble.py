"""
Evaluate Ensemble Recommender

Compare ensemble performance against individual models.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.evaluate_model import (
    compute_metrics,
    create_user_movie_mappings,
    load_test_data,
    prepare_ground_truth,
)
from src.models.ensemble import create_ensemble

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_single_model(
    embeddings_dir: str,
    user_indices: list,
    ground_truth: list,
    relevance_scores: list,
    n_items: int,
    k_values: list,
    model_name: str,
) -> Dict[str, float]:
    """Evaluate a single model."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'=' * 60}")

    # Import here to avoid circular dependency
    from src.evaluation.evaluate_model import (
        load_embeddings,
        get_top_k_recommendations,
    )

    # Load embeddings
    user_emb, movie_emb = load_embeddings(Path(embeddings_dir))

    # Generate recommendations
    predictions = get_top_k_recommendations(
        user_emb, movie_emb, user_indices, k=max(k_values)
    )

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth, relevance_scores, n_items, k_values)

    return metrics


def evaluate_ensemble_model(
    model_configs: Dict[str, str],
    weights: Dict[str, float],
    combination_method: str,
    user_indices: list,
    ground_truth: list,
    relevance_scores: list,
    n_items: int,
    k_values: list,
) -> Dict[str, float]:
    """Evaluate ensemble model."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Evaluating: ENSEMBLE")
    logger.info(f"Method: {combination_method}")
    logger.info(f"Weights: {weights}")
    logger.info(f"{'=' * 60}")

    # Create ensemble
    ensemble = create_ensemble(model_configs, weights, combination_method)

    # Generate recommendations
    predictions, _ = ensemble.get_recommendations(user_indices, k=max(k_values))

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth, relevance_scores, n_items, k_values)

    return metrics


def compare_models(results: Dict[str, Dict[str, float]], k_values: list) -> pd.DataFrame:
    """Create comparison table."""
    rows = []

    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for k in k_values:
            row[f"Recall@{k}"] = f"{metrics[f'recall@{k}']:.4f}"
            row[f"NDCG@{k}"] = f"{metrics[f'ndcg@{k}']:.4f}"
            row[f"Hit Rate@{k}"] = f"{metrics[f'hit_rate@{k}']:.4f}"
        row["Coverage"] = f"{metrics['coverage']:.4f}"
        rows.append(row)

    return pd.DataFrame(rows)


def print_results(results: Dict[str, Dict[str, float]], k_values: list) -> None:
    """Print formatted results."""
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 80)

    # Create comparison table
    df = compare_models(results, k_values)
    logger.info("\n" + df.to_string(index=False))

    # Highlight best performers
    logger.info("\n" + "=" * 80)
    logger.info("BEST PERFORMERS")
    logger.info("=" * 80)

    for k in k_values:
        # Best NDCG
        ndcg_key = f"ndcg@{k}"
        best_ndcg_model = max(results.items(), key=lambda x: x[1][ndcg_key])
        logger.info(
            f"Best NDCG@{k}: {best_ndcg_model[0]} "
            f"({best_ndcg_model[1][ndcg_key]:.4f})"
        )

        # Best Recall
        recall_key = f"recall@{k}"
        best_recall_model = max(results.items(), key=lambda x: x[1][recall_key])
        logger.info(
            f"Best Recall@{k}: {best_recall_model[0]} "
            f"({best_recall_model[1][recall_key]:.4f})"
        )

    # Best Coverage
    best_coverage_model = max(results.items(), key=lambda x: x[1]["coverage"])
    logger.info(
        f"Best Coverage: {best_coverage_model[0]} "
        f"({best_coverage_model[1]['coverage']:.4f})"
    )

    # Compute improvements
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE IMPROVEMENTS")
    logger.info("=" * 80)

    if "ENSEMBLE" in results:
        ensemble_metrics = results["ENSEMBLE"]

        # Compare to best individual model
        individual_models = {k: v for k, v in results.items() if k != "ENSEMBLE"}

        for metric_name in ["ndcg@10", "recall@10", "hit_rate@10", "coverage"]:
            best_individual = max(individual_models.values(), key=lambda x: x[metric_name])
            ensemble_value = ensemble_metrics[metric_name]
            best_individual_value = best_individual[metric_name]

            if best_individual_value > 0:
                improvement = (
                    (ensemble_value - best_individual_value) / best_individual_value * 100
                )
                logger.info(
                    f"{metric_name}: {ensemble_value:.4f} vs {best_individual_value:.4f} "
                    f"({improvement:+.1f}%)"
                )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ensemble Recommender")

    # Data paths
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
        help="Path to train ratings",
    )

    # Model configurations
    parser.add_argument(
        "--rtx4090-dir",
        type=str,
        default="data/embeddings_optimized",
        help="RTX 4090 embeddings directory",
    )
    parser.add_argument(
        "--h200-dir",
        type=str,
        default="data/embeddings_h200",
        help="H200 embeddings directory",
    )

    # Ensemble configuration
    parser.add_argument(
        "--rtx-weight",
        type=float,
        default=0.6,
        help="Weight for RTX 4090 model (default: 0.6)",
    )
    parser.add_argument(
        "--h200-weight",
        type=float,
        default=0.4,
        help="Weight for H200 model (default: 0.4)",
    )
    parser.add_argument(
        "--combination-method",
        type=str,
        default="weighted_score",
        choices=["weighted_score", "weighted_rank", "borda_count"],
        help="Ensemble combination method",
    )

    # Evaluation parameters
    parser.add_argument(
        "--rating-threshold",
        type=float,
        default=4.0,
        help="Minimum rating for relevance",
    )
    parser.add_argument(
        "--top-k", type=int, default=100, help="Generate top-K recommendations"
    )

    args = parser.parse_args()

    # Load test data
    test_df = load_test_data(Path(args.test_data))
    train_df = pd.read_parquet(Path(args.train_data))

    # Create mappings
    user_to_idx, movie_to_idx = create_user_movie_mappings(test_df, train_df)

    # Prepare ground truth
    user_indices, ground_truth, relevance_scores = prepare_ground_truth(
        test_df, user_to_idx, movie_to_idx, args.rating_threshold
    )

    k_values = [5, 10, 20, 50]
    results = {}

    # Evaluate RTX 4090 model
    if Path(args.rtx4090_dir).exists():
        results["RTX 4090"] = evaluate_single_model(
            args.rtx4090_dir,
            user_indices,
            ground_truth,
            relevance_scores,
            len(movie_to_idx),
            k_values,
            "RTX 4090",
        )

    # Evaluate H200 model
    if Path(args.h200_dir).exists():
        results["H200"] = evaluate_single_model(
            args.h200_dir,
            user_indices,
            ground_truth,
            relevance_scores,
            len(movie_to_idx),
            k_values,
            "H200",
        )

    # Evaluate ensemble
    model_configs = {}
    weights = {}

    if Path(args.rtx4090_dir).exists():
        model_configs["rtx4090"] = args.rtx4090_dir
        weights["rtx4090"] = args.rtx_weight

    if Path(args.h200_dir).exists():
        model_configs["h200"] = args.h200_dir
        weights["h200"] = args.h200_weight

    if len(model_configs) >= 2:
        results["ENSEMBLE"] = evaluate_ensemble_model(
            model_configs,
            weights,
            args.combination_method,
            user_indices,
            ground_truth,
            relevance_scores,
            len(movie_to_idx),
            k_values,
        )

        # Analyze contributions
        ensemble = create_ensemble(model_configs, weights, args.combination_method)
        contributions = ensemble.analyze_model_contributions(
            user_indices[:100], k=10  # Sample 100 users
        )

    # Print results
    print_results(results, k_values)

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
