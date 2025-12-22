"""
Unit tests for Hard Negative Sampler.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.training.hard_negative_sampler import HardNegativeSampler


class TestHardNegativeSamplerInit:
    """Test HardNegativeSampler initialization."""

    def test_init_default_weights(self):
        """Test initialization with default weights."""
        with patch("src.training.hard_negative_sampler.pd.read_parquet") as mock_read:
            ratings_df = pd.DataFrame({"userId": [1], "movieId": [1], "rating": [4.0]})
            movies_df = pd.DataFrame(
                {"movieId": [1], "title": ["Test"], "genres": ["Action"]}
            )
            mock_read.side_effect = [ratings_df, movies_df]

            sampler = HardNegativeSampler(
                train_ratings_path="dummy.parquet",
                movies_path="dummy.parquet",
                n_movies=100,
            )

            assert sampler.n_movies == 100
            assert sampler.popularity_weight == 0.7
            assert sampler.genre_weight == 0.2
            assert sampler.random_weight == 0.1
            assert sampler.popularity_top_k == 1000

    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        with patch("src.training.hard_negative_sampler.pd.read_parquet") as mock_read:
            ratings_df = pd.DataFrame({"userId": [1], "movieId": [1], "rating": [4.0]})
            movies_df = pd.DataFrame(
                {"movieId": [1], "title": ["Test"], "genres": ["Action"]}
            )
            mock_read.side_effect = [ratings_df, movies_df]

            sampler = HardNegativeSampler(
                train_ratings_path="dummy.parquet",
                movies_path="dummy.parquet",
                n_movies=100,
                popularity_weight=0.5,
                genre_weight=0.3,
                random_weight=0.2,
                popularity_top_k=500,
            )

            assert sampler.popularity_weight == 0.5
            assert sampler.genre_weight == 0.3
            assert sampler.random_weight == 0.2
            assert sampler.popularity_top_k == 500


class TestLoadData:
    """Test data loading functionality."""

    def test_load_data(self):
        """Test that data is loaded from parquet files."""
        with patch("src.training.hard_negative_sampler.pd.read_parquet") as mock_read:
            ratings_df = pd.DataFrame({"userId": [1], "movieId": [1], "rating": [4.0]})
            movies_df = pd.DataFrame(
                {"movieId": [1], "title": ["Test"], "genres": ["Action"]}
            )
            mock_read.side_effect = [ratings_df, movies_df]

            sampler = HardNegativeSampler(
                train_ratings_path="ratings.parquet",
                movies_path="movies.parquet",
                n_movies=100,
            )

            assert mock_read.call_count == 2
            pd.testing.assert_frame_equal(sampler.ratings_df, ratings_df)
            pd.testing.assert_frame_equal(sampler.movies_df, movies_df)


class TestComputeStatistics:
    """Test statistics computation methods."""

    def create_sampler_with_data(self):
        """Create sampler with mocked data loading."""
        with patch("src.training.hard_negative_sampler.pd.read_parquet") as mock_read:
            ratings_df = pd.DataFrame(
                {
                    "userId": [1, 1, 1, 2, 2, 3],
                    "movieId": [10, 20, 30, 10, 40, 20],
                    "rating": [4.0, 5.0, 3.0, 4.5, 3.5, 5.0],
                }
            )
            movies_df = pd.DataFrame(
                {
                    "movieId": [10, 20, 30, 40, 50],
                    "title": ["M10", "M20", "M30", "M40", "M50"],
                    "genres": [
                        "Action|Adv",
                        "Comedy",
                        "Action",
                        "Drama",
                        "(no genres listed)",
                    ],
                }
            )
            mock_read.side_effect = [ratings_df, movies_df]

            return HardNegativeSampler(
                train_ratings_path="ratings.parquet",
                movies_path="movies.parquet",
                n_movies=100,
            )

    def test_compute_item_popularity(self):
        """Test item popularity computation."""
        sampler = self.create_sampler_with_data()

        # Movie 10 has 2 ratings
        assert sampler.item_popularity[10] == 2
        # Movie 20 has 2 ratings
        assert sampler.item_popularity[20] == 2
        # Movie 30 has 1 rating
        assert sampler.item_popularity[30] == 1

    def test_compute_user_rated_items(self):
        """Test user rated items computation."""
        sampler = self.create_sampler_with_data()

        assert sampler.user_rated_items[1] == {10, 20, 30}
        assert sampler.user_rated_items[2] == {10, 40}
        assert sampler.user_rated_items[3] == {20}

    def test_compute_movie_genres(self):
        """Test movie genres parsing."""
        sampler = self.create_sampler_with_data()

        assert sampler.movie_genres[10] == ["Action", "Adv"]
        assert sampler.movie_genres[20] == ["Comedy"]
        assert sampler.movie_genres[50] == []  # no genres listed

    def test_compute_user_genre_preferences(self):
        """Test user genre preferences computation."""
        sampler = self.create_sampler_with_data()

        # User 1 rated Action|Adv, Comedy, Action movies
        assert "Action" in sampler.user_genre_preferences[1]
        assert "Comedy" in sampler.user_genre_preferences[1]

        # User 2 rated Action|Adv and Drama
        assert "Action" in sampler.user_genre_preferences[2]
        assert "Drama" in sampler.user_genre_preferences[2]


class TestSamplingMethods:
    """Test individual sampling methods."""

    @pytest.fixture
    def sampler(self):
        """Create sampler with mocked data."""
        with patch("src.training.hard_negative_sampler.pd.read_parquet") as mock_read:
            ratings_df = pd.DataFrame(
                {
                    "userId": [1, 1, 2, 2, 2],
                    "movieId": [0, 1, 0, 2, 3],
                    "rating": [4.0, 5.0, 4.5, 3.5, 4.0],
                }
            )
            movies_df = pd.DataFrame(
                {
                    "movieId": list(range(20)),
                    "title": [f"M{i}" for i in range(20)],
                    "genres": ["Action"] * 10 + ["Comedy"] * 10,
                }
            )
            mock_read.side_effect = [ratings_df, movies_df]

            return HardNegativeSampler(
                train_ratings_path="ratings.parquet",
                movies_path="movies.parquet",
                n_movies=20,
                popularity_top_k=10,
            )

    def test_sample_returns_correct_count(self, sampler):
        """Test that sample returns correct number of negatives."""
        negatives = sampler.sample(user_id=1, n_negatives=10)

        assert len(negatives) == 10
        assert isinstance(negatives, np.ndarray)

    def test_sample_excludes_rated_items(self, sampler):
        """Test that sampled negatives don't include rated items."""
        negatives = sampler.sample(user_id=1, n_negatives=10)
        rated_items = sampler.user_rated_items[1]

        for neg in negatives:
            assert neg not in rated_items

    def test_sample_unique_items(self, sampler):
        """Test that sampled negatives are unique."""
        # Set seed for reproducibility
        np.random.seed(42)
        negatives = sampler.sample(user_id=1, n_negatives=10)

        assert len(negatives) == len(set(negatives))

    def test_sample_within_valid_range(self, sampler):
        """Test that all sampled IDs are valid."""
        negatives = sampler.sample(user_id=1, n_negatives=10)

        assert all(0 <= neg < sampler.n_movies for neg in negatives)

    def test_sample_unknown_user(self, sampler):
        """Test sampling for user not in training data."""
        negatives = sampler.sample(user_id=999, n_negatives=10)

        # Should still return valid negatives
        assert len(negatives) == 10


class TestPrivateSamplingMethods:
    """Test private sampling helper methods."""

    @pytest.fixture
    def sampler(self):
        """Create sampler with controlled data."""
        with patch("src.training.hard_negative_sampler.pd.read_parquet") as mock_read:
            # Create ratings where movie 0 is most popular
            ratings_data = {"userId": [], "movieId": [], "rating": []}
            for i in range(50):
                ratings_data["userId"].append(i)
                ratings_data["movieId"].append(0)  # Movie 0 very popular
                ratings_data["rating"].append(4.0)
            for i in range(20):
                ratings_data["userId"].append(i)
                ratings_data["movieId"].append(1)  # Movie 1 somewhat popular
                ratings_data["rating"].append(4.0)

            ratings_df = pd.DataFrame(ratings_data)
            movies_df = pd.DataFrame(
                {
                    "movieId": list(range(30)),
                    "title": [f"M{i}" for i in range(30)],
                    "genres": ["Action"] * 15 + ["Comedy"] * 15,
                }
            )
            mock_read.side_effect = [ratings_df, movies_df]

            return HardNegativeSampler(
                train_ratings_path="ratings.parquet",
                movies_path="movies.parquet",
                n_movies=30,
                popularity_top_k=10,
            )

    def test_sample_popular_negatives(self, sampler):
        """Test popular negative sampling."""
        rated_items = {0}  # Exclude most popular

        popular_negs = sampler._sample_popular_negatives(
            user_id=0, rated_items=rated_items, n_samples=5
        )

        assert len(popular_negs) <= 5
        assert 0 not in popular_negs  # Excluded

    def test_sample_popular_negatives_empty(self, sampler):
        """Test when all popular items are rated."""
        # Rate all items in popularity_top_k
        rated_items = set(range(sampler.popularity_top_k + 5))

        popular_negs = sampler._sample_popular_negatives(
            user_id=0, rated_items=rated_items, n_samples=5
        )

        assert len(popular_negs) == 0

    def test_sample_genre_negatives(self, sampler):
        """Test genre-based negative sampling."""
        rated_items = {0, 1, 2}

        genre_negs = sampler._sample_genre_negatives(
            user_id=0, rated_items=rated_items, n_samples=5
        )

        for neg in genre_negs:
            assert neg not in rated_items

    def test_sample_genre_negatives_no_preferences(self, sampler):
        """Test genre sampling when user has no preferences."""
        sampler.user_genre_preferences[999] = set()

        genre_negs = sampler._sample_genre_negatives(
            user_id=999, rated_items=set(), n_samples=5
        )

        assert genre_negs == []

    def test_sample_random_negatives(self, sampler):
        """Test random negative sampling."""
        rated_items = {0, 1, 2, 3, 4}

        random_negs = sampler._sample_random_negatives(
            rated_items=rated_items, n_samples=10
        )

        assert len(random_negs) == 10
        for neg in random_negs:
            assert neg not in rated_items


class TestEdgeCases:
    """Test edge cases."""

    def test_no_genres_handling(self):
        """Test handling of movies without genres."""
        with patch("src.training.hard_negative_sampler.pd.read_parquet") as mock_read:
            ratings_df = pd.DataFrame(
                {
                    "userId": [1],
                    "movieId": [0],
                    "rating": [4.0],
                }
            )
            movies_df = pd.DataFrame(
                {
                    "movieId": [0, 1, 2],
                    "title": ["M0", "M1", "M2"],
                    "genres": ["(no genres listed)", None, "Action"],
                }
            )
            mock_read.side_effect = [ratings_df, movies_df]

            sampler = HardNegativeSampler(
                train_ratings_path="ratings.parquet",
                movies_path="movies.parquet",
                n_movies=10,
            )

            assert sampler.movie_genres[0] == []
            assert sampler.movie_genres[1] == []
            assert sampler.movie_genres[2] == ["Action"]

    def test_popular_items_ordering(self):
        """Test that popular items are correctly ordered."""
        with patch("src.training.hard_negative_sampler.pd.read_parquet") as mock_read:
            ratings_df = pd.DataFrame(
                {
                    "userId": [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
                    "movieId": [1, 2, 3, 1, 2, 1, 2, 3, 4, 5],
                    "rating": [4.0] * 10,
                }
            )
            movies_df = pd.DataFrame(
                {
                    "movieId": [1, 2, 3, 4, 5],
                    "title": ["M1", "M2", "M3", "M4", "M5"],
                    "genres": ["Action"] * 5,
                }
            )
            mock_read.side_effect = [ratings_df, movies_df]

            sampler = HardNegativeSampler(
                train_ratings_path="ratings.parquet",
                movies_path="movies.parquet",
                n_movies=10,
            )

            # Movie 1 and 2 have 3 ratings each (most popular)
            # Movie 3 has 2 ratings
            assert sampler.item_popularity[1] == 3
            assert sampler.item_popularity[2] == 3
            assert sampler.item_popularity[3] == 2
