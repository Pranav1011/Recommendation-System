"""
MovieLens Feature Engineering

Generates user and item features for the Two-Tower recommendation model.
Extracts interaction patterns, preferences, and content features.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Engineer features for recommendation model training."""

    def __init__(self, data_path: Path):
        """
        Initialize feature engineering.

        Args:
            data_path: Path to processed data directory
        """
        self.data_path = data_path
        self.train_df = None
        self.test_df = None
        self.movies_df = None
        self.tags_df = None

    def load_data(self) -> None:
        """Load processed data from Parquet files."""
        logger.info(f"Loading processed data from {self.data_path}")

        # Load ratings
        train_path = self.data_path / "train_ratings.parquet"
        test_path = self.data_path / "test_ratings.parquet"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "Train/test files not found. Run processor.py first"
            )

        self.train_df = pd.read_parquet(train_path)
        self.test_df = pd.read_parquet(test_path)

        logger.info(f"Loaded {len(self.train_df):,} train ratings")
        logger.info(f"Loaded {len(self.test_df):,} test ratings")

        # Load movies
        movies_path = self.data_path / "movies.parquet"
        if movies_path.exists():
            self.movies_df = pd.read_parquet(movies_path)
            logger.info(f"Loaded {len(self.movies_df):,} movies")

        # Load tags (optional)
        tags_path = self.data_path / "tags.parquet"
        if tags_path.exists():
            self.tags_df = pd.read_parquet(tags_path)
            logger.info(f"Loaded {len(self.tags_df):,} tags")

    def extract_year_from_title(self, title: str) -> int:
        """
        Extract release year from movie title.

        Args:
            title: Movie title (e.g., "Toy Story (1995)")

        Returns:
            Release year or 0 if not found
        """
        match = re.search(r"\((\d{4})\)", title)
        return int(match.group(1)) if match else 0

    def parse_genres(self, genres_str: str) -> List[str]:
        """
        Parse pipe-separated genres.

        Args:
            genres_str: Pipe-separated genre string

        Returns:
            List of genres
        """
        if (
            pd.isna(genres_str)
            or genres_str == "(no genres listed)"
            or genres_str == ""
        ):
            return []
        return genres_str.split("|")

    def engineer_movie_features(self) -> pd.DataFrame:
        """
        Engineer item (movie) features.

        Returns:
            DataFrame with movie features
        """
        logger.info("Engineering movie features...")

        # Start with movies catalog
        movie_features = self.movies_df.copy()

        # Extract year from title
        movie_features["year"] = movie_features["title"].apply(
            self.extract_year_from_title
        )

        # Parse genres
        movie_features["genre_list"] = movie_features["genres"].apply(self.parse_genres)
        movie_features["n_genres"] = movie_features["genre_list"].apply(len)

        # Compute rating statistics from training data
        rating_stats = (
            self.train_df.groupby("movieId")
            .agg(
                popularity=("rating", "count"),
                avg_rating=("rating", "mean"),
                rating_std=("rating", "std"),
                rating_min=("rating", "min"),
                rating_max=("rating", "max"),
            )
            .reset_index()
        )

        # Fill NaN std with 0 (movies with single rating)
        rating_stats["rating_std"] = rating_stats["rating_std"].fillna(0)

        # Merge with movie features
        movie_features = movie_features.merge(rating_stats, on="movieId", how="left")

        # Fill missing values (movies not in train set)
        movie_features["popularity"] = (
            movie_features["popularity"].fillna(0).astype(int)
        )
        movie_features["avg_rating"] = movie_features["avg_rating"].fillna(0)
        movie_features["rating_std"] = movie_features["rating_std"].fillna(0)
        movie_features["rating_min"] = movie_features["rating_min"].fillna(0)
        movie_features["rating_max"] = movie_features["rating_max"].fillna(0)

        # Add percentile features
        movie_features["popularity_percentile"] = movie_features["popularity"].rank(
            pct=True
        )

        # Add recency (years since release)
        current_year = 2024
        movie_features["years_since_release"] = current_year - movie_features["year"]
        movie_features["years_since_release"] = movie_features[
            "years_since_release"
        ].clip(lower=0)

        logger.info(f"Engineered features for {len(movie_features):,} movies")

        return movie_features

    def engineer_user_features(self) -> pd.DataFrame:
        """
        Engineer user features.

        Returns:
            DataFrame with user features
        """
        logger.info("Engineering user features...")

        # Basic rating statistics
        user_stats = (
            self.train_df.groupby("userId")
            .agg(
                rating_count=("rating", "count"),
                avg_rating=("rating", "mean"),
                rating_std=("rating", "std"),
                rating_min=("rating", "min"),
                rating_max=("rating", "max"),
            )
            .reset_index()
        )

        # Fill NaN std with 0
        user_stats["rating_std"] = user_stats["rating_std"].fillna(0)

        # Compute temporal features
        temporal_stats = (
            self.train_df.groupby("userId")["timestamp"]
            .agg(["min", "max", "mean"])
            .reset_index()
        )
        temporal_stats.columns = [
            "userId",
            "first_rating_time",
            "last_rating_time",
            "avg_rating_time",
        ]

        # Merge temporal features
        user_features = user_stats.merge(temporal_stats, on="userId")

        # Compute activity span (in days)
        user_features["activity_span_days"] = (
            user_features["last_rating_time"] - user_features["first_rating_time"]
        ) / (24 * 3600)
        user_features["activity_span_days"] = user_features["activity_span_days"].clip(
            lower=0
        )

        # Compute rating frequency (ratings per day)
        user_features["ratings_per_day"] = user_features["rating_count"] / (
            user_features["activity_span_days"] + 1
        )  # +1 to avoid division by zero

        # Genre preferences
        genre_prefs = self._compute_genre_preferences()
        user_features = user_features.merge(genre_prefs, on="userId", how="left")

        # Fill missing genre preferences with 0
        genre_cols = [col for col in user_features.columns if col.startswith("genre_")]
        user_features[genre_cols] = user_features[genre_cols].fillna(0)

        logger.info(f"Engineered features for {len(user_features):,} users")

        return user_features

    def _compute_genre_preferences(self) -> pd.DataFrame:
        """
        Compute genre preference scores for each user.

        Returns:
            DataFrame with userId and genre preference columns
        """
        logger.info("Computing genre preferences...")

        # Merge ratings with movie genres
        ratings_genres = self.train_df.merge(
            self.movies_df[["movieId", "genres"]], on="movieId"
        )

        # Parse genres
        ratings_genres["genre_list"] = ratings_genres["genres"].apply(self.parse_genres)

        # Explode genre list (one row per genre per rating)
        genre_ratings = ratings_genres.explode("genre_list")
        genre_ratings = genre_ratings[genre_ratings["genre_list"] != ""]

        # Compute average rating per user per genre
        genre_prefs = (
            genre_ratings.groupby(["userId", "genre_list"])["rating"]
            .mean()
            .reset_index()
        )

        # Pivot to wide format
        genre_prefs_wide = genre_prefs.pivot(
            index="userId", columns="genre_list", values="rating"
        )

        # Rename columns with 'genre_' prefix
        genre_prefs_wide.columns = [
            f"genre_{col.lower().replace('-', '_')}" for col in genre_prefs_wide.columns
        ]

        genre_prefs_wide = genre_prefs_wide.reset_index()

        logger.info(f"Computed {len(genre_prefs_wide.columns) - 1} genre preferences")

        return genre_prefs_wide

    def create_genre_encoding(self) -> Dict[str, int]:
        """
        Create genre to index mapping for embedding layer.

        Returns:
            Dictionary mapping genre names to indices
        """
        logger.info("Creating genre encoding...")

        # Get all unique genres
        all_genres = set()
        for genres_str in self.movies_df["genres"]:
            genres = self.parse_genres(genres_str)
            all_genres.update(genres)

        # Sort for consistent ordering
        all_genres = sorted(all_genres)

        # Create mapping (0 is reserved for padding/unknown)
        genre_to_idx = {genre: idx + 1 for idx, genre in enumerate(all_genres)}
        genre_to_idx["unknown"] = 0

        logger.info(f"Created encoding for {len(genre_to_idx)} genres")

        return genre_to_idx

    def save_features(
        self,
        movie_features: pd.DataFrame,
        user_features: pd.DataFrame,
        genre_encoding: Dict[str, int],
        output_dir: Path,
    ) -> None:
        """
        Save engineered features to disk.

        Args:
            movie_features: Movie feature DataFrame
            user_features: User feature DataFrame
            genre_encoding: Genre to index mapping
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving features to {output_dir}")

        # Save movie features
        movie_path = output_dir / "movie_features.parquet"
        movie_features.to_parquet(movie_path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved movie features to {movie_path}")

        # Save user features
        user_path = output_dir / "user_features.parquet"
        user_features.to_parquet(user_path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved user features to {user_path}")

        # Save genre encoding
        genre_path = output_dir / "genre_encoding.json"
        with open(genre_path, "w") as f:
            json.dump(genre_encoding, f, indent=2)
        logger.info(f"Saved genre encoding to {genre_path}")

        # Save feature statistics
        stats = {
            "n_movies": len(movie_features),
            "n_users": len(user_features),
            "n_genres": len(genre_encoding),
            "movie_features": {
                "columns": list(movie_features.columns),
                "n_columns": len(movie_features.columns),
            },
            "user_features": {
                "columns": list(user_features.columns),
                "n_columns": len(user_features.columns),
            },
        }

        stats_path = output_dir / "feature_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved feature statistics to {stats_path}")

    def process(self, output_dir: Path) -> None:
        """
        Execute full feature engineering pipeline.

        Args:
            output_dir: Directory to save features
        """
        self.load_data()

        movie_features = self.engineer_movie_features()
        user_features = self.engineer_user_features()
        genre_encoding = self.create_genre_encoding()

        self.save_features(movie_features, user_features, genre_encoding, output_dir)


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Engineer features from MovieLens data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed",
        help="Input directory with processed data (default: data/processed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/features",
        help="Output directory for features (default: data/features)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.error("Run processor.py first to create processed data")
        exit(1)

    try:
        engineer = FeatureEngineering(input_dir)
        engineer.process(output_dir)

        logger.info("\nFeature engineering complete!")
        logger.info(f"Features saved to: {output_dir}")
        logger.info("\nNext steps:")
        logger.info("  1. Explore features: jupyter notebook notebooks/01_eda.ipynb")
        logger.info("  2. Train model: python src/training/train_two_tower.py")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
