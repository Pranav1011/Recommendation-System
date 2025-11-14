"""
PyTorch Dataset for MovieLens Recommendation System

Implements efficient data loading for training the Two-Tower model.
Supports both explicit ratings and negative sampling for BPR loss.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MovieLensDataset(Dataset):
    """
    PyTorch Dataset for MovieLens ratings.

    Loads ratings and features from Parquet files.
    Supports negative sampling for BPR loss.
    """

    def __init__(
        self,
        ratings_path: Path,
        user_features_path: Optional[Path] = None,
        movie_features_path: Optional[Path] = None,
        negative_sampling: bool = False,
        n_negatives: int = 4,
        hard_negative_sampler=None,
    ):
        """
        Initialize MovieLens Dataset.

        Args:
            ratings_path: Path to ratings Parquet file
            user_features_path: Path to user features Parquet file
            movie_features_path: Path to movie features Parquet file
            negative_sampling: Whether to sample negative items
            n_negatives: Number of negative samples per positive sample
            hard_negative_sampler: HardNegativeSampler instance (optional)
        """
        self.ratings_path = ratings_path
        self.user_features_path = user_features_path
        self.movie_features_path = movie_features_path
        self.negative_sampling = negative_sampling
        self.n_negatives = n_negatives
        self.hard_negative_sampler = hard_negative_sampler

        # Load data
        self._load_data()

    def _load_data(self) -> None:
        """Load ratings and features from Parquet files."""
        logger.info(f"Loading ratings from {self.ratings_path}")
        self.ratings_df = pd.read_parquet(self.ratings_path)

        logger.info(f"Loaded {len(self.ratings_df):,} ratings")

        # Create user ID to index mapping
        self.unique_users = sorted(self.ratings_df["userId"].unique())
        self.unique_movies = sorted(self.ratings_df["movieId"].unique())

        self.user_to_idx = {
            user_id: idx for idx, user_id in enumerate(self.unique_users)
        }
        self.movie_to_idx = {
            movie_id: idx for idx, movie_id in enumerate(self.unique_movies)
        }

        # Map IDs to indices in ratings dataframe
        self.ratings_df["user_idx"] = self.ratings_df["userId"].map(self.user_to_idx)
        self.ratings_df["movie_idx"] = self.ratings_df["movieId"].map(self.movie_to_idx)

        # Load user features if provided
        self.user_features = None
        if self.user_features_path and self.user_features_path.exists():
            logger.info(f"Loading user features from {self.user_features_path}")
            user_features_df = pd.read_parquet(self.user_features_path)

            # Select numeric features only
            numeric_cols = user_features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "userId"]

            # Create feature matrix (aligned with user_to_idx)
            self.user_features = np.zeros(
                (len(self.unique_users), len(numeric_cols)), dtype=np.float32
            )

            for idx, user_id in enumerate(self.unique_users):
                user_row = user_features_df[user_features_df["userId"] == user_id]
                if len(user_row) > 0:
                    self.user_features[idx] = user_row[numeric_cols].values[0]

            logger.info(f"Loaded user features: {self.user_features.shape[1]} features")

        # Load movie features if provided
        self.movie_features = None
        if self.movie_features_path and self.movie_features_path.exists():
            logger.info(f"Loading movie features from {self.movie_features_path}")
            movie_features_df = pd.read_parquet(self.movie_features_path)

            # Select numeric features only
            numeric_cols = movie_features_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "movieId"]

            # Create feature matrix (aligned with movie_to_idx)
            self.movie_features = np.zeros(
                (len(self.unique_movies), len(numeric_cols)), dtype=np.float32
            )

            for idx, movie_id in enumerate(self.unique_movies):
                movie_row = movie_features_df[movie_features_df["movieId"] == movie_id]
                if len(movie_row) > 0:
                    self.movie_features[idx] = movie_row[numeric_cols].values[0]

            logger.info(
                f"Loaded movie features: {self.movie_features.shape[1]} features"
            )

        # For negative sampling: track movies rated by each user
        if self.negative_sampling:
            self.user_positive_movies = (
                self.ratings_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
            )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.ratings_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - user_id: User index
                - movie_id: Movie index
                - rating: Rating value
                - user_features: User features (if available)
                - movie_features: Movie features (if available)
                - neg_movie_ids: Negative movie indices (if negative_sampling=True)
        """
        row = self.ratings_df.iloc[idx]

        # Convert indices to Python int to avoid numpy indexing issues
        user_idx = int(row["user_idx"])
        movie_idx = int(row["movie_idx"])

        sample = {
            "user_id": torch.tensor(user_idx, dtype=torch.long),
            "movie_id": torch.tensor(movie_idx, dtype=torch.long),
            "rating": torch.tensor(row["rating"], dtype=torch.float32),
        }

        # Add user features
        if self.user_features is not None:
            sample["user_features"] = torch.tensor(
                self.user_features[user_idx], dtype=torch.float32
            )

        # Add movie features
        if self.movie_features is not None:
            sample["movie_features"] = torch.tensor(
                self.movie_features[movie_idx], dtype=torch.float32
            )

        # Negative sampling for BPR loss
        if self.negative_sampling:
            # Use hard negative sampler if provided
            if self.hard_negative_sampler is not None:
                # Get the original user_id (not the index)
                user_id = int(row["userId"])
                neg_movie_indices = self.hard_negative_sampler.sample(
                    user_id, n_negatives=self.n_negatives
                )
            else:
                # Fall back to random negative sampling
                positive_movies = self.user_positive_movies.get(user_idx, set())

                # Sample negative movies (movies NOT rated by this user)
                all_movies = set(range(len(self.unique_movies)))
                negative_candidates = list(all_movies - positive_movies)

                if len(negative_candidates) >= self.n_negatives:
                    neg_movie_indices = np.random.choice(
                        negative_candidates, size=self.n_negatives, replace=False
                    )
                else:
                    # If not enough negative candidates, sample with replacement
                    neg_movie_indices = np.random.choice(
                        negative_candidates, size=self.n_negatives, replace=True
                    )

            sample["neg_movie_ids"] = torch.tensor(neg_movie_indices, dtype=torch.long)

        return sample

    def get_num_users(self) -> int:
        """Get number of unique users."""
        return len(self.unique_users)

    def get_num_movies(self) -> int:
        """Get number of unique movies."""
        return len(self.unique_movies)

    def get_feature_dims(self) -> Tuple[int, int]:
        """
        Get feature dimensions.

        Returns:
            Tuple of (user_feature_dim, movie_feature_dim)
        """
        user_dim = self.user_features.shape[1] if self.user_features is not None else 0
        movie_dim = (
            self.movie_features.shape[1] if self.movie_features is not None else 0
        )
        return user_dim, movie_dim


def create_dataloaders(
    train_ratings_path: Path,
    test_ratings_path: Path,
    user_features_path: Optional[Path] = None,
    movie_features_path: Optional[Path] = None,
    batch_size: int = 512,
    num_workers: int = 4,
    negative_sampling: bool = False,
    n_negatives: int = 4,
    hard_negative_sampler=None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test dataloaders.

    Args:
        train_ratings_path: Path to train ratings
        test_ratings_path: Path to test ratings
        user_features_path: Path to user features
        movie_features_path: Path to movie features
        batch_size: Batch size
        num_workers: Number of data loading workers
        negative_sampling: Whether to use negative sampling
        n_negatives: Number of negative samples
        hard_negative_sampler: HardNegativeSampler instance (optional)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = MovieLensDataset(
        ratings_path=train_ratings_path,
        user_features_path=user_features_path,
        movie_features_path=movie_features_path,
        negative_sampling=negative_sampling,
        n_negatives=n_negatives,
        hard_negative_sampler=hard_negative_sampler,
    )

    test_dataset = MovieLensDataset(
        ratings_path=test_ratings_path,
        user_features_path=user_features_path,
        movie_features_path=movie_features_path,
        negative_sampling=False,  # No negative sampling for evaluation
    )

    # Determine if we should use pin_memory (not supported on MPS)
    use_pin_memory = torch.cuda.is_available()

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    logger.info(f"Created dataloaders:")
    logger.info(
        f"  - Train: {len(train_dataset):,} samples, {len(train_loader)} batches"
    )
    logger.info(f"  - Test: {len(test_dataset):,} samples, {len(test_loader)} batches")

    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing MovieLens Dataset...")

    data_dir = Path("data/processed")
    features_dir = Path("data/features")

    train_path = data_dir / "train_ratings.parquet"
    test_path = data_dir / "test_ratings.parquet"
    user_features_path = features_dir / "user_features.parquet"
    movie_features_path = features_dir / "movie_features.parquet"

    if train_path.exists():
        # Test dataset creation
        dataset = MovieLensDataset(
            ratings_path=train_path,
            user_features_path=user_features_path,
            movie_features_path=movie_features_path,
            negative_sampling=True,
            n_negatives=4,
        )

        print(f"✓ Dataset created successfully")
        print(f"  - Number of samples: {len(dataset):,}")
        print(f"  - Number of users: {dataset.get_num_users():,}")
        print(f"  - Number of movies: {dataset.get_num_movies():,}")

        user_dim, movie_dim = dataset.get_feature_dims()
        print(f"  - User feature dim: {user_dim}")
        print(f"  - Movie feature dim: {movie_dim}")

        # Test getting a sample
        sample = dataset[0]
        print(f"\n✓ Sample structure:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")

        # Test dataloader creation
        if test_path.exists():
            train_loader, test_loader = create_dataloaders(
                train_ratings_path=train_path,
                test_ratings_path=test_path,
                user_features_path=user_features_path,
                movie_features_path=movie_features_path,
                batch_size=512,
                num_workers=0,  # 0 for testing
                negative_sampling=True,
            )

            # Test batch loading
            batch = next(iter(train_loader))
            print(f"\n✓ Batch structure:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")

    else:
        print(f"✗ Train data not found at {train_path}")
        print("  Run data processing first: python src/data/processor.py")
