"""
Hard Negative Sampler for BPR Training

Samples challenging negatives instead of random ones:
1. Popularity-based: Popular items user didn't interact with
2. Genre-based: Items from same genres user ignored
3. Random: For diversity
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HardNegativeSampler:
    """
    Sample hard negatives for BPR training.

    Hard negatives are more challenging than random negatives:
    - Popular items user didn't rate (model must learn nuanced preferences)
    - Same-genre items user ignored (model must learn fine-grained taste)
    - Random items (for diversity)
    """

    def __init__(
        self,
        train_ratings_path: Path,
        movies_path: Path,
        n_movies: int,
        popularity_weight: float = 0.7,
        genre_weight: float = 0.2,
        random_weight: float = 0.1,
        popularity_top_k: int = 1000,
    ):
        """
        Initialize Hard Negative Sampler.

        Args:
            train_ratings_path: Path to training ratings parquet
            movies_path: Path to movies parquet with genres
            n_movies: Total number of movies in the dataset
            popularity_weight: Weight for popularity-based negatives (default 0.7)
            genre_weight: Weight for genre-based negatives (default 0.2)
            random_weight: Weight for random negatives (default 0.1)
            popularity_top_k: Sample from top K popular items (default 1000)
        """
        self.n_movies = n_movies
        self.popularity_weight = popularity_weight
        self.genre_weight = genre_weight
        self.random_weight = random_weight
        self.popularity_top_k = popularity_top_k

        logger.info("Initializing Hard Negative Sampler...")
        logger.info(f"  Popularity weight: {popularity_weight}")
        logger.info(f"  Genre weight: {genre_weight}")
        logger.info(f"  Random weight: {random_weight}")

        # Load data
        self._load_data(train_ratings_path, movies_path)

        # Pre-compute statistics
        self._compute_item_popularity()
        self._compute_user_rated_items()
        self._compute_movie_genres()
        self._compute_user_genre_preferences()

        logger.info("Hard Negative Sampler initialized!")
        logger.info(f"  Total movies: {self.n_movies}")
        logger.info(f"  Total users: {len(self.user_rated_items)}")
        logger.info(f"  Avg ratings per user: {np.mean([len(items) for items in self.user_rated_items.values()]):.1f}")

    def _load_data(self, train_ratings_path: Path, movies_path: Path):
        """Load training ratings and movies."""
        logger.info("Loading training data...")
        self.ratings_df = pd.read_parquet(train_ratings_path)
        self.movies_df = pd.read_parquet(movies_path)
        logger.info(f"  Loaded {len(self.ratings_df):,} ratings")
        logger.info(f"  Loaded {len(self.movies_df):,} movies")

    def _compute_item_popularity(self):
        """Compute item popularity (rating counts)."""
        logger.info("Computing item popularity...")
        popularity = self.ratings_df.groupby('movieId').size().to_dict()

        # Sort by popularity
        self.popular_items = sorted(
            popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Convert to movie_id -> rank mapping
        self.item_popularity = {
            movie_id: count for movie_id, count in self.popular_items
        }

        logger.info(f"  Most popular movie has {self.popular_items[0][1]:,} ratings")
        logger.info(f"  Median popularity: {np.median(list(self.item_popularity.values())):.0f} ratings")

    def _compute_user_rated_items(self):
        """Compute set of items each user has rated."""
        logger.info("Computing user rated items...")
        self.user_rated_items = (
            self.ratings_df.groupby('userId')['movieId']
            .apply(set)
            .to_dict()
        )

    def _compute_movie_genres(self):
        """Extract genres for each movie."""
        logger.info("Extracting movie genres...")

        # Parse genres (format: "Action|Adventure|Sci-Fi")
        def parse_genres(genre_str):
            if pd.isna(genre_str) or genre_str == "(no genres listed)":
                return []
            return genre_str.split('|')

        self.movie_genres = {}
        for _, row in self.movies_df.iterrows():
            movie_id = row['movieId']
            genres = parse_genres(row['genres'])
            self.movie_genres[movie_id] = genres

        # Count unique genres
        all_genres = set()
        for genres in self.movie_genres.values():
            all_genres.update(genres)
        logger.info(f"  Found {len(all_genres)} unique genres")

    def _compute_user_genre_preferences(self):
        """Compute which genres each user has interacted with."""
        logger.info("Computing user genre preferences...")

        self.user_genre_preferences = {}
        for user_id, rated_items in self.user_rated_items.items():
            user_genres = set()
            for movie_id in rated_items:
                if movie_id in self.movie_genres:
                    user_genres.update(self.movie_genres[movie_id])
            self.user_genre_preferences[user_id] = user_genres

        avg_genres = np.mean([len(g) for g in self.user_genre_preferences.values()])
        logger.info(f"  Avg genres per user: {avg_genres:.1f}")

    def sample(self, user_id: int, n_negatives: int = 100) -> np.ndarray:
        """
        Sample hard negatives for a user.

        Args:
            user_id: User ID to sample negatives for
            n_negatives: Number of negatives to sample (default 100)

        Returns:
            Array of movie IDs (negatives)
        """
        # Get user's rated items
        rated_items = self.user_rated_items.get(user_id, set())

        # Compute how many of each type to sample
        n_popular = int(n_negatives * self.popularity_weight)
        n_genre = int(n_negatives * self.genre_weight)
        n_random = n_negatives - n_popular - n_genre

        negatives = []

        # 1. Sample popular negatives
        popular_negatives = self._sample_popular_negatives(
            user_id, rated_items, n_popular
        )
        negatives.extend(popular_negatives)

        # 2. Sample genre-based negatives
        genre_negatives = self._sample_genre_negatives(
            user_id, rated_items, n_genre
        )
        negatives.extend(genre_negatives)

        # 3. Sample random negatives
        random_negatives = self._sample_random_negatives(
            rated_items, n_random
        )
        negatives.extend(random_negatives)

        # Ensure we have exactly n_negatives (pad with random if needed)
        while len(negatives) < n_negatives:
            candidate = np.random.randint(0, self.n_movies)
            if candidate not in rated_items and candidate not in negatives:
                negatives.append(candidate)

        return np.array(negatives[:n_negatives])

    def _sample_popular_negatives(
        self,
        user_id: int,
        rated_items: Set[int],
        n_samples: int
    ) -> List[int]:
        """Sample popular items user didn't rate."""
        # Get top K popular items user hasn't rated
        candidates = [
            movie_id for movie_id, _ in self.popular_items[:self.popularity_top_k]
            if movie_id not in rated_items
        ]

        if len(candidates) == 0:
            return []

        # Sample with probability proportional to popularity
        # (more popular = higher chance of being sampled)
        weights = np.array([
            self.item_popularity[movie_id] for movie_id in candidates
        ])
        weights = weights / weights.sum()

        n_samples = min(n_samples, len(candidates))
        sampled = np.random.choice(
            candidates,
            size=n_samples,
            replace=False,
            p=weights
        )

        return sampled.tolist()

    def _sample_genre_negatives(
        self,
        user_id: int,
        rated_items: Set[int],
        n_samples: int
    ) -> List[int]:
        """Sample items from same genres user has rated."""
        # Get user's preferred genres
        user_genres = self.user_genre_preferences.get(user_id, set())

        if len(user_genres) == 0:
            return []

        # Find movies in those genres that user hasn't rated
        candidates = []
        for movie_id, genres in self.movie_genres.items():
            if movie_id not in rated_items:
                # Check if movie has any of user's preferred genres
                if any(g in user_genres for g in genres):
                    candidates.append(movie_id)

        if len(candidates) == 0:
            return []

        n_samples = min(n_samples, len(candidates))
        sampled = np.random.choice(candidates, size=n_samples, replace=False)

        return sampled.tolist()

    def _sample_random_negatives(
        self,
        rated_items: Set[int],
        n_samples: int
    ) -> List[int]:
        """Sample random items for diversity."""
        negatives = []

        while len(negatives) < n_samples:
            candidate = np.random.randint(0, self.n_movies)
            if candidate not in rated_items and candidate not in negatives:
                negatives.append(candidate)

        return negatives
