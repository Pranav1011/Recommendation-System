"""
MovieLens Data Processor

Loads, cleans, and prepares MovieLens data for model training.
Creates train/test splits and saves as Parquet files for efficient loading.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MovieLensProcessor:
    """Processes MovieLens datasets for recommendation system training."""

    def __init__(self, data_path: Path, dataset_size: str):
        """
        Initialize processor.

        Args:
            data_path: Path to raw dataset directory
            dataset_size: Dataset size (25m, 1m, or latest-small)
        """
        self.data_path = data_path
        self.dataset_size = dataset_size
        self.ratings_df = None
        self.movies_df = None
        self.links_df = None
        self.tags_df = None

        # Determine file format based on dataset size
        self.file_format = "dat" if dataset_size == "1m" else "csv"
        self.separator = "::" if dataset_size == "1m" else ","

    def load_data(self) -> None:
        """
        Load all dataset files into memory.

        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If data validation fails
        """
        logger.info(f"Loading {self.dataset_size} dataset from {self.data_path}")

        # Load ratings
        ratings_file = self.data_path / f"ratings.{self.file_format}"
        if not ratings_file.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_file}")

        if self.file_format == "dat":
            self.ratings_df = pd.read_csv(
                ratings_file,
                sep=self.separator,
                engine="python",
                names=["userId", "movieId", "rating", "timestamp"],
                dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
            )
        else:
            self.ratings_df = pd.read_csv(
                ratings_file,
                dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
            )

        logger.info(f"Loaded {len(self.ratings_df):,} ratings")

        # Load movies
        movies_file = self.data_path / f"movies.{self.file_format}"
        if not movies_file.exists():
            raise FileNotFoundError(f"Movies file not found: {movies_file}")

        if self.file_format == "dat":
            # 1M dataset has different encoding
            self.movies_df = pd.read_csv(
                movies_file,
                sep=self.separator,
                engine="python",
                names=["movieId", "title", "genres"],
                encoding="latin-1",
            )
        else:
            self.movies_df = pd.read_csv(movies_file)

        logger.info(f"Loaded {len(self.movies_df):,} movies")

        # Load links (not in 1M dataset)
        if self.dataset_size != "1m":
            links_file = self.data_path / "links.csv"
            if links_file.exists():
                self.links_df = pd.read_csv(links_file)
                logger.info(f"Loaded {len(self.links_df):,} links")

        # Load tags (optional)
        if self.dataset_size != "1m":
            tags_file = self.data_path / "tags.csv"
            if tags_file.exists():
                self.tags_df = pd.read_csv(tags_file)
                logger.info(f"Loaded {len(self.tags_df):,} tags")
        else:
            tags_file = self.data_path / "tags.dat"
            if tags_file.exists():
                self.tags_df = pd.read_csv(
                    tags_file,
                    sep=self.separator,
                    engine="python",
                    names=["userId", "movieId", "tag", "timestamp"],
                    encoding="latin-1",
                )
                logger.info(f"Loaded {len(self.tags_df):,} tags")

    def validate_data(self) -> None:
        """
        Validate loaded data for consistency and quality.

        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating data...")

        # Check for missing values
        null_ratings = self.ratings_df.isnull().sum().sum()
        if null_ratings > 0:
            logger.warning(f"Found {null_ratings} null values in ratings")

        # Check rating range
        min_rating = self.ratings_df["rating"].min()
        max_rating = self.ratings_df["rating"].max()
        logger.info(f"Rating range: {min_rating} - {max_rating}")

        if min_rating < 0 or max_rating > 5:
            raise ValueError(f"Invalid rating range: {min_rating} - {max_rating}")

        # Check for duplicate ratings
        duplicates = self.ratings_df.duplicated(subset=["userId", "movieId"]).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate user-movie pairs")
            logger.info("Keeping last rating for duplicates")
            self.ratings_df = self.ratings_df.drop_duplicates(
                subset=["userId", "movieId"], keep="last"
            )

        # Validate movie IDs match
        rating_movies = set(self.ratings_df["movieId"].unique())
        catalog_movies = set(self.movies_df["movieId"].unique())
        missing_movies = rating_movies - catalog_movies

        if missing_movies:
            logger.warning(
                f"Found {len(missing_movies)} movie IDs in ratings not in catalog"
            )
            logger.info("Filtering out ratings for missing movies")
            self.ratings_df = self.ratings_df[
                self.ratings_df["movieId"].isin(catalog_movies)
            ]

        logger.info("Data validation complete")

    def create_train_test_split(
        self, test_size: float = 0.2, temporal: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split from ratings data.

        Args:
            test_size: Fraction of data to use for testing (default: 0.2)
            temporal: Use temporal split (last 20% of interactions) vs random

        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(
            f"Creating {'temporal' if temporal else 'random'} train/test split..."
        )

        if temporal:
            # Sort by timestamp and split
            sorted_ratings = self.ratings_df.sort_values("timestamp")
            split_idx = int(len(sorted_ratings) * (1 - test_size))

            train_df = sorted_ratings.iloc[:split_idx].copy()
            test_df = sorted_ratings.iloc[split_idx:].copy()
        else:
            # Random split
            test_df = self.ratings_df.sample(frac=test_size, random_state=42)
            train_df = self.ratings_df.drop(test_df.index)

        logger.info(f"Train set: {len(train_df):,} ratings")
        logger.info(f"Test set: {len(test_df):,} ratings")

        return train_df, test_df

    def compute_statistics(self, train_df: pd.DataFrame) -> Dict:
        """
        Compute dataset statistics.

        Args:
            train_df: Training data

        Returns:
            Dictionary of statistics
        """
        logger.info("Computing dataset statistics...")

        stats = {
            "n_users": train_df["userId"].nunique(),
            "n_movies": train_df["movieId"].nunique(),
            "n_ratings": len(train_df),
            "sparsity": 1
            - (
                len(train_df)
                / (train_df["userId"].nunique() * train_df["movieId"].nunique())
            ),
            "avg_rating": train_df["rating"].mean(),
            "rating_std": train_df["rating"].std(),
            "ratings_per_user": {
                "mean": train_df.groupby("userId").size().mean(),
                "median": train_df.groupby("userId").size().median(),
                "min": train_df.groupby("userId").size().min(),
                "max": train_df.groupby("userId").size().max(),
            },
            "ratings_per_movie": {
                "mean": train_df.groupby("movieId").size().mean(),
                "median": train_df.groupby("movieId").size().median(),
                "min": train_df.groupby("movieId").size().min(),
                "max": train_df.groupby("movieId").size().max(),
            },
        }

        logger.info(f"Users: {stats['n_users']:,}")
        logger.info(f"Movies: {stats['n_movies']:,}")
        logger.info(f"Ratings: {stats['n_ratings']:,}")
        logger.info(f"Sparsity: {stats['sparsity']:.4%}")
        logger.info(
            f"Avg Rating: {stats['avg_rating']:.2f} ± {stats['rating_std']:.2f}"
        )

        return stats

    def save_to_parquet(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
        stats: Dict,
    ) -> None:
        """
        Save processed data to Parquet files.

        Args:
            train_df: Training data
            test_df: Test data
            output_dir: Output directory
            stats: Dataset statistics to save
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving processed data to {output_dir}")

        # Save ratings
        train_path = output_dir / "train_ratings.parquet"
        test_path = output_dir / "test_ratings.parquet"

        train_df.to_parquet(train_path, engine="pyarrow", compression="snappy")
        test_df.to_parquet(test_path, engine="pyarrow", compression="snappy")

        logger.info(f"Saved train ratings to {train_path}")
        logger.info(f"Saved test ratings to {test_path}")

        # Save movies catalog
        movies_path = output_dir / "movies.parquet"
        self.movies_df.to_parquet(movies_path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved movies catalog to {movies_path}")

        # Save links if available
        if self.links_df is not None:
            links_path = output_dir / "links.parquet"
            self.links_df.to_parquet(links_path, engine="pyarrow", compression="snappy")
            logger.info(f"Saved links to {links_path}")

        # Save tags if available
        if self.tags_df is not None:
            tags_path = output_dir / "tags.parquet"
            self.tags_df.to_parquet(tags_path, engine="pyarrow", compression="snappy")
            logger.info(f"Saved tags to {tags_path}")

        # Save statistics
        stats_path = output_dir / "statistics.json"
        import json

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native_types(obj):
            """Recursively convert numpy types to Python native types."""
            import numpy as np

            if isinstance(obj, dict):
                return {
                    key: convert_to_native_types(value) for key, value in obj.items()
                }
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        stats_native = convert_to_native_types(stats)

        with open(stats_path, "w") as f:
            json.dump(stats_native, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")

    def process(
        self, output_dir: Path, test_size: float = 0.2, temporal: bool = True
    ) -> Dict:
        """
        Execute full processing pipeline.

        Args:
            output_dir: Directory to save processed data
            test_size: Fraction of data for testing
            temporal: Use temporal split

        Returns:
            Dataset statistics
        """
        self.load_data()
        self.validate_data()

        train_df, test_df = self.create_train_test_split(
            test_size=test_size, temporal=temporal
        )
        stats = self.compute_statistics(train_df)

        self.save_to_parquet(train_df, test_df, output_dir, stats)

        return stats


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Process MovieLens dataset")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing raw dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for processed data (default: data/processed)",
    )
    parser.add_argument(
        "--dataset-size",
        type=str,
        choices=["25m", "1m", "latest-small"],
        default="latest-small",
        help="Dataset size (default: latest-small)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)",
    )
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Use random split instead of temporal",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        exit(1)

    try:
        processor = MovieLensProcessor(input_dir, args.dataset_size)
        processor.process(
            output_dir, test_size=args.test_size, temporal=not args.random_split
        )

        logger.info("\nProcessing complete!")
        logger.info(f"Processed data saved to: {output_dir}")
        logger.info("\nNext steps:")
        logger.info("  1. Explore data: jupyter notebook notebooks/01_eda.ipynb")
        logger.info("  2. Feature engineering: python src/data/feature_engineering.py")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
