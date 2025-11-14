"""
Comprehensive Ensemble Model Evaluation

Evaluates ensemble recommender on test set and compares with individual models.
Computes all ranking metrics and generates detailed reports.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

matplotlib.use("Agg")  # Use non-interactive backend

from src.models.ensemble_recommender import EnsembleRecommender  # noqa: E402
from src.training.metrics import compute_all_metrics  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def load_test_data(
    test_path: Path, user_to_idx: Dict, item_to_idx: Dict, min_rating: float = 4.0
) -> Tuple[List[int], List[Set[int]], List[Dict[int, float]]]:
    """
    Load and prepare test data.

    Args:
        test_path: Path to test ratings
        user_to_idx: User ID to index mapping
        item_to_idx: Item ID to index mapping
        min_rating: Minimum rating to consider as relevant

    Returns:
        Tuple of (user_indices, ground_truth_sets, relevance_scores)
    """
    logger.info(f"Loading test data from {test_path}")

    # Load test ratings
    test_df = pd.read_parquet(test_path)
    logger.info(f"Loaded {len(test_df):,} test ratings")

    # Filter for users/items in our index
    test_df = test_df[
        test_df["userId"].isin(user_to_idx.keys())
        & test_df["movieId"].isin(item_to_idx.keys())
    ].copy()

    logger.info(f"After filtering: {len(test_df):,} ratings")

    # Map to indices
    test_df["user_idx"] = test_df["userId"].map(user_to_idx)
    test_df["item_idx"] = test_df["movieId"].map(item_to_idx)

    # Group by user
    user_groups = test_df.groupby("user_idx")

    user_indices = []
    ground_truth = []
    relevance_scores = []

    for user_idx, group in user_groups:
        user_indices.append(user_idx)

        # Ground truth: items with rating >= threshold
        relevant_items = set(group[group["rating"] >= min_rating]["item_idx"].tolist())
        ground_truth.append(relevant_items)

        # Relevance scores: use normalized ratings
        scores = dict(zip(group["item_idx"], group["rating"]))
        relevance_scores.append(scores)

    logger.info(f"Prepared {len(user_indices)} test users")
    logger.info(
        f"Avg relevant items per user: {np.mean([len(gt) for gt in ground_truth]):.1f}"
    )

    return user_indices, ground_truth, relevance_scores


def generate_predictions(
    ensemble: EnsembleRecommender,
    user_indices: List[int],
    k: int = 100,
    batch_size: int = 1000,
) -> List[List[int]]:
    """
    Generate top-K predictions for users.

    Args:
        ensemble: Ensemble recommender
        user_indices: List of user indices
        k: Number of recommendations per user
        batch_size: Batch size for processing

    Returns:
        List of top-K item indices for each user
    """
    logger.info(f"Generating top-{k} predictions for {len(user_indices)} users...")

    predictions = []

    for i in tqdm(range(0, len(user_indices), batch_size), desc="Batches"):
        batch_user_indices = user_indices[i : i + batch_size]

        # Get user embeddings
        user_emb = ensemble.user_embeddings[batch_user_indices]

        # Compute similarity with all items
        # (batch_size, n_items)
        similarities = torch.matmul(user_emb, ensemble.item_embeddings.T)

        # Get top-K for each user
        _, top_k_indices = torch.topk(similarities, k, dim=1)

        predictions.extend(top_k_indices.cpu().numpy().tolist())

    return predictions


def evaluate_ensemble(
    ensemble: EnsembleRecommender,
    test_path: Path,
    k_values: List[int] = [5, 10, 20, 50],
    min_rating: float = 4.0,
) -> Dict[str, float]:
    """
    Evaluate ensemble model on test set.

    Args:
        ensemble: Ensemble recommender
        test_path: Path to test ratings
        k_values: List of K values for metrics
        min_rating: Minimum rating threshold

    Returns:
        Dictionary of metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE MODEL EVALUATION")
    logger.info("=" * 80)

    # Load test data
    user_indices, ground_truth, relevance_scores = load_test_data(
        test_path, ensemble.user_to_idx, ensemble.item_to_idx, min_rating
    )

    # Generate predictions
    max_k = max(k_values)
    predictions = generate_predictions(ensemble, user_indices, k=max_k)

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(
        predictions,
        ground_truth,
        relevance_scores,
        n_items=len(ensemble.item_to_idx),
        k_values=k_values,
    )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    for k in k_values:
        logger.info(f"\nTop-{k} Metrics:")
        logger.info(
            f"  Recall@{k}:    {metrics[f'recall@{k}']:.4f} ({metrics[f'recall@{k}']*100:.2f}%)"
        )
        logger.info(
            f"  Precision@{k}: {metrics[f'precision@{k}']:.4f} ({metrics[f'precision@{k}']*100:.2f}%)"
        )
        logger.info(
            f"  NDCG@{k}:      {metrics[f'ndcg@{k}']:.4f} ({metrics[f'ndcg@{k}']*100:.2f}%)"
        )
        logger.info(
            f"  Hit Rate@{k}:  {metrics[f'hit_rate@{k}']:.4f} ({metrics[f'hit_rate@{k}']*100:.2f}%)"
        )
        logger.info(
            f"  MAP@{k}:       {metrics[f'map@{k}']:.4f} ({metrics[f'map@{k}']*100:.2f}%)"
        )

    logger.info(
        f"\nCoverage: {metrics['coverage']:.4f} ({metrics['coverage']*100:.2f}%)"
    )
    logger.info("=" * 80)

    return metrics


def load_individual_model_results(results_dir: Path) -> Dict[str, Dict]:
    """
    Load evaluation results from individual models.

    Args:
        results_dir: Directory containing evaluation results

    Returns:
        Dictionary mapping model name to metrics
    """
    logger.info("Loading individual model results...")

    models_results = {}

    # LightGCN results
    lightgcn_results = results_dir / "evaluation_results.json"
    if lightgcn_results.exists():
        with open(lightgcn_results, "r") as f:
            models_results["LightGCN"] = json.load(f)
        logger.info("  - LightGCN results loaded")

    # Two-Tower results (from embeddings_optimized)
    twotower_results_dir = Path("results/twotower_evaluation")
    if twotower_results_dir.exists():
        twotower_results = twotower_results_dir / "metrics.json"
        if twotower_results.exists():
            with open(twotower_results, "r") as f:
                models_results["Two-Tower"] = json.load(f)
            logger.info("  - Two-Tower results loaded")

    return models_results


def compare_models(
    ensemble_metrics: Dict[str, float],
    individual_results: Dict[str, Dict],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Compare ensemble with individual models.

    Args:
        ensemble_metrics: Ensemble metrics
        individual_results: Individual model metrics
        output_dir: Output directory for plots

    Returns:
        Comparison dataframe
    """
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)

    # Combine all results
    all_results = {"Ensemble": ensemble_metrics, **individual_results}

    # Extract key metrics at K=10
    comparison_data = []
    for model_name, metrics in all_results.items():
        comparison_data.append(
            {
                "Model": model_name,
                "Recall@10": metrics.get("recall@10", 0),
                "Precision@10": metrics.get("precision@10", 0),
                "NDCG@10": metrics.get("ndcg@10", 0),
                "Hit Rate@10": metrics.get("hit_rate@10", 0),
                "MAP@10": metrics.get("map@10", 0),
                "Coverage": metrics.get("coverage", 0),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    # Print comparison table
    logger.info("\n" + comparison_df.to_string(index=False))

    # Calculate improvements
    if "LightGCN" in all_results:
        lightgcn_recall = all_results["LightGCN"].get("recall@10", 0)
        ensemble_recall = ensemble_metrics.get("recall@10", 0)
        if lightgcn_recall > 0:
            improvement = ((ensemble_recall - lightgcn_recall) / lightgcn_recall) * 100
            logger.info(f"\nEnsemble vs LightGCN (Recall@10): {improvement:+.2f}%")

    if "Two-Tower" in all_results:
        twotower_recall = all_results["Two-Tower"].get("recall@10", 0)
        ensemble_recall = ensemble_metrics.get("recall@10", 0)
        if twotower_recall > 0:
            improvement = ((ensemble_recall - twotower_recall) / twotower_recall) * 100
            logger.info(f"Ensemble vs Two-Tower (Recall@10): {improvement:+.2f}%")

    logger.info("=" * 80)

    # Save comparison
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"\nComparison saved to {comparison_path}")

    return comparison_df


def plot_metrics_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot metrics comparison across models.

    Args:
        comparison_df: Comparison dataframe
        output_dir: Output directory
    """
    logger.info("Generating comparison plots...")

    # Metrics to plot (exclude Coverage for separate plot)
    metrics = ["Recall@10", "Precision@10", "NDCG@10", "Hit Rate@10", "MAP@10"]

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = comparison_df[["Model", metric]].sort_values(metric, ascending=False)

        bars = ax.bar(data["Model"], data[metric], color=colors[: len(data)])
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} Comparison", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Coverage plot
    ax = axes[5]
    data = comparison_df[["Model", "Coverage"]].sort_values("Coverage", ascending=False)
    bars = ax.bar(data["Model"], data["Coverage"], color=colors[: len(data)])
    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_title("Coverage Comparison", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "metrics_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Comparison plot saved to {plot_path}")
    plt.close()


def generate_markdown_report(
    ensemble_metrics: Dict[str, float],
    comparison_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
) -> None:
    """
    Generate markdown evaluation report.

    Args:
        ensemble_metrics: Ensemble metrics
        comparison_df: Model comparison dataframe
        output_dir: Output directory
        config: Evaluation configuration
    """
    logger.info("Generating markdown report...")

    report_lines = [
        "# Ensemble Model Evaluation Report",
        "",
        "## Configuration",
        "",
        f"- **LightGCN Weight**: {config['lightgcn_weight']:.2f}",
        f"- **Two-Tower Weight**: {config['twotower_weight']:.2f}",
        f"- **Rating Threshold**: {config['min_rating']:.1f}",
        f"- **Evaluation K Values**: {config['k_values']}",
        "",
        "## Ensemble Results",
        "",
        "### Top-10 Metrics",
        "",
        "| Metric | Value | Percentage |",
        "|--------|-------|------------|",
        (
            f"| Recall@10 | {ensemble_metrics.get('recall@10', 0):.4f} | "
            f"{ensemble_metrics.get('recall@10', 0)*100:.2f}% |"
        ),
        (
            f"| Precision@10 | {ensemble_metrics.get('precision@10', 0):.4f} | "
            f"{ensemble_metrics.get('precision@10', 0)*100:.2f}% |"
        ),
        (
            f"| NDCG@10 | {ensemble_metrics.get('ndcg@10', 0):.4f} | "
            f"{ensemble_metrics.get('ndcg@10', 0)*100:.2f}% |"
        ),
        (
            f"| Hit Rate@10 | {ensemble_metrics.get('hit_rate@10', 0):.4f} | "
            f"{ensemble_metrics.get('hit_rate@10', 0)*100:.2f}% |"
        ),
        (
            f"| MAP@10 | {ensemble_metrics.get('map@10', 0):.4f} | "
            f"{ensemble_metrics.get('map@10', 0)*100:.2f}% |"
        ),
        "",
        (
            f"**Coverage**: {ensemble_metrics.get('coverage', 0):.4f} "
            f"({ensemble_metrics.get('coverage', 0)*100:.2f}%)"
        ),
        "",
        "## Model Comparison",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
    ]

    # Add interpretation
    recall_10 = ensemble_metrics.get("recall@10", 0)
    ndcg_10 = ensemble_metrics.get("ndcg@10", 0)

    if recall_10 >= 0.20:
        report_lines.append(
            "- **Recall@10**: EXCELLENT (>= 0.20) - Industry standard achieved"
        )
    elif recall_10 >= 0.15:
        report_lines.append("- **Recall@10**: GOOD (>= 0.15) - Acceptable performance")
    elif recall_10 >= 0.10:
        report_lines.append("- **Recall@10**: FAIR (>= 0.10) - Room for improvement")
    else:
        report_lines.append(
            "- **Recall@10**: POOR (< 0.10) - Needs significant improvement"
        )

    if ndcg_10 >= 0.25:
        report_lines.append(
            "- **NDCG@10**: EXCELLENT (>= 0.25) - Strong ranking quality"
        )
    elif ndcg_10 >= 0.20:
        report_lines.append("- **NDCG@10**: GOOD (>= 0.20) - Decent ranking quality")
    elif ndcg_10 >= 0.15:
        report_lines.append("- **NDCG@10**: FAIR (>= 0.15) - Weak ranking quality")
    else:
        report_lines.append("- **NDCG@10**: POOR (< 0.15) - Needs improvement")

    report_lines.extend(
        [
            "",
            "## Visualizations",
            "",
            "![Metrics Comparison](metrics_comparison.png)",
            "",
        ]
    )

    # Save report
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble recommender")
    parser.add_argument(
        "--lightgcn-checkpoint",
        type=str,
        default="models/checkpoints/best_model_lightgcn_optimized.pt",
        help="Path to LightGCN checkpoint",
    )
    parser.add_argument(
        "--lightgcn-config",
        type=str,
        default="configs/train_config_lightgcn_optimized.json",
        help="Path to LightGCN config",
    )
    parser.add_argument(
        "--twotower-checkpoint",
        type=str,
        default="models/checkpoints/best_model_optimized.pt",
        help="Path to Two-Tower checkpoint",
    )
    parser.add_argument(
        "--twotower-config",
        type=str,
        default="configs/train_config_bpr_optimized.json",
        help="Path to Two-Tower config",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test_ratings.parquet",
        help="Path to test ratings",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="models/checkpoints",
        help="Directory containing individual model results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ensemble_evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--lightgcn-weight",
        type=float,
        default=0.7,
        help="Weight for LightGCN",
    )
    parser.add_argument(
        "--twotower-weight",
        type=float,
        default=0.3,
        help="Weight for Two-Tower",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=4.0,
        help="Minimum rating threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("ENSEMBLE MODEL EVALUATION")
    logger.info("=" * 80)

    # Initialize ensemble
    logger.info("Initializing ensemble recommender...")
    ensemble = EnsembleRecommender(
        lightgcn_checkpoint=args.lightgcn_checkpoint,
        lightgcn_config=args.lightgcn_config,
        twotower_checkpoint=args.twotower_checkpoint,
        twotower_config=args.twotower_config,
        lightgcn_weight=args.lightgcn_weight,
        twotower_weight=args.twotower_weight,
        device=args.device,
    )

    # Evaluate ensemble
    k_values = [5, 10, 20, 50]
    ensemble_metrics = evaluate_ensemble(
        ensemble,
        Path(args.test_data),
        k_values=k_values,
        min_rating=args.min_rating,
    )

    # Save ensemble results
    results_path = output_dir / "ensemble_metrics.json"
    with open(results_path, "w") as f:
        json.dump(ensemble_metrics, f, indent=2)
    logger.info(f"\nEnsemble metrics saved to {results_path}")

    # Load individual model results and compare
    individual_results = load_individual_model_results(Path(args.results_dir))

    if individual_results:
        comparison_df = compare_models(ensemble_metrics, individual_results, output_dir)
        plot_metrics_comparison(comparison_df, output_dir)
    else:
        logger.warning("No individual model results found for comparison")
        comparison_df = pd.DataFrame(
            [
                {
                    "Model": "Ensemble",
                    "Recall@10": ensemble_metrics.get("recall@10", 0),
                    "Precision@10": ensemble_metrics.get("precision@10", 0),
                    "NDCG@10": ensemble_metrics.get("ndcg@10", 0),
                    "Hit Rate@10": ensemble_metrics.get("hit_rate@10", 0),
                    "MAP@10": ensemble_metrics.get("map@10", 0),
                    "Coverage": ensemble_metrics.get("coverage", 0),
                }
            ]
        )

    # Generate report
    config = {
        "lightgcn_weight": args.lightgcn_weight,
        "twotower_weight": args.twotower_weight,
        "min_rating": args.min_rating,
        "k_values": k_values,
    }
    generate_markdown_report(ensemble_metrics, comparison_df, output_dir, config)

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir.absolute()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
