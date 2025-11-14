"""
Generate Ensemble Embeddings

Creates ensemble embeddings by combining multiple trained models.
Saves embeddings in multiple formats for downstream use.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.ensemble_recommender import EnsembleRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_metadata(
    movies_path: Path, user_to_idx: Dict[int, int], item_to_idx: Dict[int, int]
) -> tuple:
    """
    Load movie metadata for enrichment.

    Args:
        movies_path: Path to movies.parquet
        user_to_idx: User ID to index mapping
        item_to_idx: Item ID to index mapping

    Returns:
        Tuple of (user_metadata_df, movie_metadata_df)
    """
    logger.info("Loading metadata...")

    # Load movies
    movies_df = pd.read_parquet(movies_path)
    logger.info(f"Loaded {len(movies_df):,} movies")

    # Filter to movies in our index
    movies_df = movies_df[movies_df["movieId"].isin(item_to_idx.keys())].copy()
    movies_df["movie_idx"] = movies_df["movieId"].map(item_to_idx)
    movies_df = movies_df.sort_values("movie_idx")

    # Create user metadata (minimal - just IDs)
    user_ids = sorted(user_to_idx.keys())
    user_indices = [user_to_idx[uid] for uid in user_ids]
    users_df = pd.DataFrame({"userId": user_ids, "user_idx": user_indices})

    logger.info(f"Metadata prepared: {len(users_df)} users, {len(movies_df)} movies")

    return users_df, movies_df


def save_embeddings_with_metadata(
    ensemble: EnsembleRecommender,
    users_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save embeddings with metadata as Parquet.

    Args:
        ensemble: Ensemble recommender
        users_df: User metadata
        movies_df: Movie metadata
        output_dir: Output directory
    """
    logger.info("Saving embeddings with metadata...")

    # User embeddings
    user_embeddings = ensemble.user_embeddings.cpu().numpy()
    user_emb_df = users_df.copy()

    # Add embedding columns
    for i in tqdm(range(user_embeddings.shape[1]), desc="User embedding columns"):
        user_emb_df[f"emb_{i}"] = user_embeddings[:, i]

    user_emb_path = output_dir / "user_embeddings_with_metadata.parquet"
    user_emb_df.to_parquet(user_emb_path, index=False)
    logger.info(f"Saved user embeddings to {user_emb_path}")

    # Movie embeddings
    item_embeddings = ensemble.item_embeddings.cpu().numpy()
    movie_emb_df = movies_df.copy()

    # Add embedding columns
    for i in tqdm(range(item_embeddings.shape[1]), desc="Movie embedding columns"):
        movie_emb_df[f"emb_{i}"] = item_embeddings[:, i]

    movie_emb_path = output_dir / "movie_embeddings_with_metadata.parquet"
    movie_emb_df.to_parquet(movie_emb_path, index=False)
    logger.info(f"Saved movie embeddings to {movie_emb_path}")


def save_statistics(ensemble: EnsembleRecommender, output_dir: Path) -> None:
    """
    Save embedding statistics.

    Args:
        ensemble: Ensemble recommender
        output_dir: Output directory
    """
    logger.info("Computing embedding statistics...")

    user_emb = ensemble.user_embeddings.cpu().numpy()
    item_emb = ensemble.item_embeddings.cpu().numpy()

    stats = {
        "n_users": int(user_emb.shape[0]),
        "n_items": int(item_emb.shape[0]),
        "embedding_dim": int(user_emb.shape[1]),
        "lightgcn_weight": float(ensemble.lightgcn_weight),
        "twotower_weight": float(ensemble.twotower_weight),
        "user_embedding_stats": {
            "mean": float(user_emb.mean()),
            "std": float(user_emb.std()),
            "min": float(user_emb.min()),
            "max": float(user_emb.max()),
            "l2_norm_mean": float(np.linalg.norm(user_emb, axis=1).mean()),
        },
        "item_embedding_stats": {
            "mean": float(item_emb.mean()),
            "std": float(item_emb.std()),
            "min": float(item_emb.min()),
            "max": float(item_emb.max()),
            "l2_norm_mean": float(np.linalg.norm(item_emb, axis=1).mean()),
        },
    }

    stats_path = output_dir / "embedding_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Statistics saved to {stats_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Number of users: {stats['n_users']:,}")
    logger.info(f"Number of items: {stats['n_items']:,}")
    logger.info(f"Embedding dimension: {stats['embedding_dim']}")
    logger.info(f"\nModel weights:")
    logger.info(f"  LightGCN: {stats['lightgcn_weight']:.2f}")
    logger.info(f"  Two-Tower: {stats['twotower_weight']:.2f}")
    logger.info(f"\nUser embeddings:")
    logger.info(f"  Mean: {stats['user_embedding_stats']['mean']:.4f}")
    logger.info(f"  Std: {stats['user_embedding_stats']['std']:.4f}")
    logger.info(
        f"  L2 norm (mean): {stats['user_embedding_stats']['l2_norm_mean']:.4f}"
    )
    logger.info(f"\nItem embeddings:")
    logger.info(f"  Mean: {stats['item_embedding_stats']['mean']:.4f}")
    logger.info(f"  Std: {stats['item_embedding_stats']['std']:.4f}")
    logger.info(
        f"  L2 norm (mean): {stats['item_embedding_stats']['l2_norm_mean']:.4f}"
    )
    logger.info("=" * 60)


def generate_sample_recommendations(
    ensemble: EnsembleRecommender,
    movies_df: pd.DataFrame,
    output_dir: Path,
    n_users: int = 10,
    k: int = 10,
) -> None:
    """
    Generate sample recommendations for verification.

    Args:
        ensemble: Ensemble recommender
        movies_df: Movie metadata
        output_dir: Output directory
        n_users: Number of sample users
        k: Number of recommendations per user
    """
    logger.info(f"Generating sample recommendations for {n_users} users...")

    # Get sample users
    sample_user_ids = list(ensemble.user_to_idx.keys())[:n_users]

    samples = []
    for user_id in tqdm(sample_user_ids, desc="Generating recommendations"):
        try:
            items, scores = ensemble.recommend(user_id, k=k, exclude_seen=True)

            # Get movie titles
            for item_id, score in zip(items, scores):
                movie_info = movies_df[movies_df["movieId"] == item_id]
                if not movie_info.empty:
                    title = movie_info.iloc[0]["title"]
                    genres = movie_info.iloc[0].get("genres", "")
                else:
                    title = f"Unknown (ID: {item_id})"
                    genres = ""

                samples.append(
                    {
                        "userId": user_id,
                        "movieId": item_id,
                        "title": title,
                        "genres": genres,
                        "score": score,
                    }
                )
        except Exception as e:
            logger.warning(
                f"Failed to generate recommendations for user {user_id}: {e}"
            )

    # Save as CSV for easy viewing
    samples_df = pd.DataFrame(samples)
    samples_path = output_dir / "sample_recommendations.csv"
    samples_df.to_csv(samples_path, index=False)
    logger.info(f"Sample recommendations saved to {samples_path}")

    # Print a few examples
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE RECOMMENDATIONS (First User)")
    logger.info("=" * 60)
    user_samples = samples_df[samples_df["userId"] == sample_user_ids[0]]
    for idx, row in user_samples.iterrows():
        logger.info(f"{idx+1}. {row['title']}")
        logger.info(f"   Score: {row['score']:.4f} | Genres: {row['genres']}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ensemble embeddings from trained models"
    )
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
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/features",
        help="Directory containing feature data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings_ensemble",
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--lightgcn-weight", type=float, default=0.7, help="Weight for LightGCN model"
    )
    parser.add_argument(
        "--twotower-weight", type=float, default=0.3, help="Weight for Two-Tower model"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save embeddings with metadata (Parquet format)",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate sample recommendations",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ENSEMBLE EMBEDDING GENERATION")
    logger.info("=" * 80)
    logger.info(f"LightGCN checkpoint: {args.lightgcn_checkpoint}")
    logger.info(f"Two-Tower checkpoint: {args.twotower_checkpoint}")
    logger.info(
        f"Weights: LightGCN={args.lightgcn_weight:.2f}, TwoTower={args.twotower_weight:.2f}"
    )
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ensemble
    logger.info("\nInitializing ensemble recommender...")
    ensemble = EnsembleRecommender(
        lightgcn_checkpoint=args.lightgcn_checkpoint,
        lightgcn_config=args.lightgcn_config,
        twotower_checkpoint=args.twotower_checkpoint,
        twotower_config=args.twotower_config,
        data_dir=args.data_dir,
        features_dir=args.features_dir,
        lightgcn_weight=args.lightgcn_weight,
        twotower_weight=args.twotower_weight,
        device=args.device,
    )

    # Save basic embeddings (numpy arrays)
    logger.info("\nSaving embeddings...")
    ensemble.save_embeddings(args.output_dir)

    # Save statistics
    save_statistics(ensemble, output_dir)

    # Save with metadata if requested
    if args.save_metadata:
        data_dir = Path(args.data_dir)
        users_df, movies_df = load_metadata(
            data_dir / "movies.parquet", ensemble.user_to_idx, ensemble.item_to_idx
        )
        save_embeddings_with_metadata(ensemble, users_df, movies_df, output_dir)

        # Generate sample recommendations if requested
        if args.generate_samples:
            generate_sample_recommendations(
                ensemble, movies_df, output_dir, n_users=10, k=10
            )

    logger.info("\n" + "=" * 80)
    logger.info("EMBEDDING GENERATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Embeddings saved to: {output_dir.absolute()}")
    logger.info("\nFiles created:")
    logger.info(
        f"  - user_embeddings.npy ({ensemble.user_embeddings.shape[0]:,} x {ensemble.user_embeddings.shape[1]})"
    )
    logger.info(
        f"  - item_embeddings.npy ({ensemble.item_embeddings.shape[0]:,} x {ensemble.item_embeddings.shape[1]})"
    )
    logger.info(f"  - user_to_idx.json")
    logger.info(f"  - item_to_idx.json")
    logger.info(f"  - embedding_statistics.json")
    if args.save_metadata:
        logger.info(f"  - user_embeddings_with_metadata.parquet")
        logger.info(f"  - movie_embeddings_with_metadata.parquet")
    if args.generate_samples:
        logger.info(f"  - sample_recommendations.csv")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
