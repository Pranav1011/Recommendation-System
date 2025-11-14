"""
Embedding Generation Script

Generates and saves user and movie embeddings from trained Two-Tower model.
Embeddings will be used for:
- Vector database indexing (Qdrant)
- Similarity search
- Recommendation retrieval
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate and save embeddings from trained model."""

    def __init__(
        self,
        model_path: Path,
        user_features_path: Optional[Path] = None,
        movie_features_path: Optional[Path] = None,
        device: str = "cuda",
    ):
        """
        Initialize embedding generator.

        Args:
            model_path: Path to trained model checkpoint
            user_features_path: Path to user features
            movie_features_path: Path to movie features
            device: Device to use for inference
        """
        self.model_path = model_path
        self.user_features_path = user_features_path
        self.movie_features_path = movie_features_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")

        # Load model
        self._load_model()

        # Load features
        self._load_features()

    def _load_model(self):
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Get model config from checkpoint
        config = checkpoint["config"]

        # Create model
        from src.models.two_tower import create_model

        self.model = create_model(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.n_users = config["n_users"]
        self.n_movies = config["n_movies"]

        logger.info("Model loaded successfully")
        logger.info(f"  - Users: {self.n_users:,}")
        logger.info(f"  - Movies: {self.n_movies:,}")

    def _load_features(self):
        """Load user and movie features."""
        self.user_features = None
        self.movie_features = None

        # Load user features
        if self.user_features_path and self.user_features_path.exists():
            logger.info(f"Loading user features from {self.user_features_path}")
            user_features_df = pd.read_parquet(self.user_features_path)

            # Select numeric features only
            numeric_cols = user_features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "userId"]

            # Create feature matrix
            self.user_features = user_features_df.sort_values("userId")[
                numeric_cols
            ].values.astype(np.float32)

            logger.info(f"Loaded user features: {self.user_features.shape}")

        # Load movie features
        if self.movie_features_path and self.movie_features_path.exists():
            logger.info(f"Loading movie features from {self.movie_features_path}")
            movie_features_df = pd.read_parquet(self.movie_features_path)

            # Select numeric features only
            numeric_cols = movie_features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "movieId"]

            # Create feature matrix
            self.movie_features = movie_features_df.sort_values("movieId")[
                numeric_cols
            ].values.astype(np.float32)

            logger.info(f"Loaded movie features: {self.movie_features.shape}")

    @torch.no_grad()
    def generate_user_embeddings(self, batch_size: int = 1024) -> np.ndarray:
        """
        Generate embeddings for all users.

        Args:
            batch_size: Batch size for inference

        Returns:
            User embeddings (n_users, embedding_dim)
        """
        logger.info(f"Generating embeddings for {self.n_users:,} users...")

        all_embeddings = []

        for start_idx in tqdm(
            range(0, self.n_users, batch_size), desc="User embeddings"
        ):
            end_idx = min(start_idx + batch_size, self.n_users)

            # Create user IDs batch
            user_ids = torch.arange(start_idx, end_idx, device=self.device)

            # Get user features if available
            user_features = None
            if self.user_features is not None:
                user_features = torch.from_numpy(
                    self.user_features[start_idx:end_idx]
                ).to(self.device)

            # Generate embeddings
            embeddings = self.model.get_user_embedding(user_ids, user_features)

            # Move to CPU and convert to numpy
            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        user_embeddings = np.vstack(all_embeddings)

        logger.info(f"Generated user embeddings: {user_embeddings.shape}")

        return user_embeddings

    @torch.no_grad()
    def generate_movie_embeddings(self, batch_size: int = 1024) -> np.ndarray:
        """
        Generate embeddings for all movies.

        Args:
            batch_size: Batch size for inference

        Returns:
            Movie embeddings (n_movies, embedding_dim)
        """
        logger.info(f"Generating embeddings for {self.n_movies:,} movies...")

        all_embeddings = []

        for start_idx in tqdm(
            range(0, self.n_movies, batch_size), desc="Movie embeddings"
        ):
            end_idx = min(start_idx + batch_size, self.n_movies)

            # Create movie IDs batch
            movie_ids = torch.arange(start_idx, end_idx, device=self.device)

            # Get movie features if available
            movie_features = None
            if self.movie_features is not None:
                movie_features = torch.from_numpy(
                    self.movie_features[start_idx:end_idx]
                ).to(self.device)

            # Generate embeddings
            embeddings = self.model.get_movie_embedding(movie_ids, movie_features)

            # Move to CPU and convert to numpy
            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        movie_embeddings = np.vstack(all_embeddings)

        logger.info(f"Generated movie embeddings: {movie_embeddings.shape}")

        return movie_embeddings

    def save_embeddings(
        self,
        user_embeddings: np.ndarray,
        movie_embeddings: np.ndarray,
        output_dir: Path,
    ):
        """
        Save embeddings to disk.

        Args:
            user_embeddings: User embeddings
            movie_embeddings: Movie embeddings
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving embeddings to {output_dir}")

        # Save as numpy arrays (fast loading)
        user_path = output_dir / "user_embeddings.npy"
        movie_path = output_dir / "movie_embeddings.npy"

        np.save(user_path, user_embeddings)
        np.save(movie_path, movie_embeddings)

        logger.info(f"Saved user embeddings to {user_path}")
        logger.info(f"Saved movie embeddings to {movie_path}")

        # Also save as Parquet with metadata
        user_df = pd.DataFrame(
            user_embeddings,
            columns=[f"emb_{i}" for i in range(user_embeddings.shape[1])],
        )
        user_df["userId"] = range(len(user_df))

        movie_df = pd.DataFrame(
            movie_embeddings,
            columns=[f"emb_{i}" for i in range(movie_embeddings.shape[1])],
        )
        movie_df["movieId"] = range(len(movie_df))

        user_parquet_path = output_dir / "user_embeddings.parquet"
        movie_parquet_path = output_dir / "movie_embeddings.parquet"

        user_df.to_parquet(user_parquet_path, engine="pyarrow", compression="snappy")
        movie_df.to_parquet(movie_parquet_path, engine="pyarrow", compression="snappy")

        logger.info(f"Saved user embeddings (Parquet) to {user_parquet_path}")
        logger.info(f"Saved movie embeddings (Parquet) to {movie_parquet_path}")

        # Save metadata
        metadata = {
            "n_users": int(self.n_users),
            "n_movies": int(self.n_movies),
            "embedding_dim": int(user_embeddings.shape[1]),
            "user_embeddings_shape": list(user_embeddings.shape),
            "movie_embeddings_shape": list(movie_embeddings.shape),
            "model_path": str(self.model_path),
        }

        import json

        metadata_path = output_dir / "embedding_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main entry point for embedding generation."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings from trained model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--user-features",
        type=str,
        default="data/features/user_features.parquet",
        help="Path to user features",
    )
    parser.add_argument(
        "--movie-features",
        type=str,
        default="data/features/movie_features.parquet",
        help="Path to movie features",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings",
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    # Create paths
    model_path = Path(args.model_path)
    user_features_path = Path(args.user_features)
    movie_features_path = Path(args.movie_features)
    output_dir = Path(args.output_dir)

    # Create generator
    generator = EmbeddingGenerator(
        model_path=model_path,
        user_features_path=user_features_path,
        movie_features_path=movie_features_path,
        device=args.device,
    )

    # Generate embeddings
    user_embeddings = generator.generate_user_embeddings(batch_size=args.batch_size)
    movie_embeddings = generator.generate_movie_embeddings(batch_size=args.batch_size)

    # Save embeddings
    generator.save_embeddings(user_embeddings, movie_embeddings, output_dir)

    logger.info("\nEmbedding generation complete!")
    logger.info(f"User embeddings: {user_embeddings.shape}")
    logger.info(f"Movie embeddings: {movie_embeddings.shape}")
    logger.info(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
