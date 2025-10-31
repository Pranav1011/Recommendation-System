"""
Tests for MovieLens Feature Engineering module.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.feature_engineering import FeatureEngineering


@pytest.fixture
def sample_ratings_train():
    """Create sample training ratings data."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3],
            "movieId": [1, 2, 3, 1, 3, 2],
            "rating": [4.0, 5.0, 3.0, 2.0, 4.0, 5.0],
            "timestamp": [1000, 2000, 3000, 1500, 2500, 3500],
        }
    )


@pytest.fixture
def sample_ratings_test():
    """Create sample test ratings data."""
    return pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [2, 1],
            "rating": [4.5, 3.0],
            "timestamp": [4000, 4500],
        }
    )


@pytest.fixture
def sample_movies():
    """Create sample movies data."""
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["Movie A (2000)", "Movie B (2010)", "Movie C (1995)"],
            "genres": ["Action|Adventure", "Comedy", "Drama|Romance"],
        }
    )


@pytest.fixture
def sample_tags():
    """Create sample tags data."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [1, 2, 1],
            "tag": ["great", "funny", "classic"],
            "timestamp": [1000, 2000, 1500],
        }
    )


@pytest.fixture
def sample_data_path(
    tmp_path, sample_ratings_train, sample_ratings_test, sample_movies, sample_tags
):
    """Create temporary directory with sample data files."""
    # Create processed directory
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Save sample data
    sample_ratings_train.to_parquet(processed_dir / "train_ratings.parquet")
    sample_ratings_test.to_parquet(processed_dir / "test_ratings.parquet")
    sample_movies.to_parquet(processed_dir / "movies.parquet")
    sample_tags.to_parquet(processed_dir / "tags.parquet")

    return processed_dir


class TestFeatureEngineering:
    """Test FeatureEngineering class."""

    def test_init(self, sample_data_path):
        """Test initialization."""
        fe = FeatureEngineering(sample_data_path)
        assert fe.data_path == sample_data_path
        assert fe.train_df is None
        assert fe.movies_df is None

    def test_load_data(self, sample_data_path):
        """Test data loading."""
        fe = FeatureEngineering(sample_data_path)
        fe.load_data()

        assert fe.train_df is not None
        assert fe.test_df is not None
        assert fe.movies_df is not None
        assert fe.tags_df is not None
        assert len(fe.train_df) == 6
        assert len(fe.test_df) == 2
        assert len(fe.movies_df) == 3

    def test_load_data_missing_files(self, tmp_path):
        """Test loading data when files are missing."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        fe = FeatureEngineering(empty_dir)
        with pytest.raises(FileNotFoundError):
            fe.load_data()

    def test_extract_year_from_title(self, sample_data_path):
        """Test year extraction from movie titles."""
        fe = FeatureEngineering(sample_data_path)

        assert fe.extract_year_from_title("Toy Story (1995)") == 1995
        assert fe.extract_year_from_title("The Matrix (1999)") == 1999
        assert fe.extract_year_from_title("No Year") == 0
        assert fe.extract_year_from_title("Invalid (abcd)") == 0

    def test_parse_genres(self, sample_data_path):
        """Test genre parsing."""
        fe = FeatureEngineering(sample_data_path)

        assert fe.parse_genres("Action|Adventure") == ["Action", "Adventure"]
        assert fe.parse_genres("Comedy") == ["Comedy"]
        assert fe.parse_genres("(no genres listed)") == []
        assert fe.parse_genres(None) == []

    def test_engineer_movie_features(self, sample_data_path):
        """Test movie feature engineering."""
        fe = FeatureEngineering(sample_data_path)
        fe.load_data()

        movie_features = fe.engineer_movie_features()

        # Check basic columns
        assert "movieId" in movie_features.columns
        assert "year" in movie_features.columns
        assert "genre_list" in movie_features.columns
        assert "n_genres" in movie_features.columns
        assert "popularity" in movie_features.columns
        assert "avg_rating" in movie_features.columns
        assert "rating_std" in movie_features.columns

        # Check values
        assert len(movie_features) == 3
        assert movie_features["year"].min() >= 0
        assert movie_features["popularity"].min() >= 0

    def test_engineer_user_features(self, sample_data_path):
        """Test user feature engineering."""
        fe = FeatureEngineering(sample_data_path)
        fe.load_data()

        user_features = fe.engineer_user_features()

        # Check basic columns
        assert "userId" in user_features.columns
        assert "rating_count" in user_features.columns
        assert "avg_rating" in user_features.columns
        assert "rating_std" in user_features.columns
        assert "first_rating_time" in user_features.columns
        assert "last_rating_time" in user_features.columns
        assert "activity_span_days" in user_features.columns
        assert "ratings_per_day" in user_features.columns

        # Check values
        assert len(user_features) == 3  # 3 users
        assert user_features["rating_count"].min() > 0
        assert user_features["activity_span_days"].min() >= 0

    def test_compute_genre_preferences(self, sample_data_path):
        """Test genre preference computation."""
        fe = FeatureEngineering(sample_data_path)
        fe.load_data()

        genre_prefs = fe._compute_genre_preferences()

        assert "userId" in genre_prefs.columns
        assert len(genre_prefs) > 0
        # Should have genre_ prefixed columns
        genre_cols = [col for col in genre_prefs.columns if col.startswith("genre_")]
        assert len(genre_cols) > 0

    def test_create_genre_encoding(self, sample_data_path):
        """Test genre encoding creation."""
        fe = FeatureEngineering(sample_data_path)
        fe.load_data()

        genre_encoding = fe.create_genre_encoding()

        assert isinstance(genre_encoding, dict)
        assert "unknown" in genre_encoding
        assert genre_encoding["unknown"] == 0
        # Should have at least 4 genres from sample data (Action, Adventure, Comedy, Drama, Romance)
        assert len(genre_encoding) >= 5

    def test_save_features(self, sample_data_path, tmp_path):
        """Test feature saving."""
        fe = FeatureEngineering(sample_data_path)
        fe.load_data()

        movie_features = fe.engineer_movie_features()
        user_features = fe.engineer_user_features()
        genre_encoding = fe.create_genre_encoding()

        output_dir = tmp_path / "features"
        fe.save_features(movie_features, user_features, genre_encoding, output_dir)

        # Check files exist
        assert (output_dir / "movie_features.parquet").exists()
        assert (output_dir / "user_features.parquet").exists()
        assert (output_dir / "genre_encoding.json").exists()
        assert (output_dir / "feature_statistics.json").exists()

        # Verify content
        with open(output_dir / "genre_encoding.json", "r") as f:
            loaded_encoding = json.load(f)
            assert loaded_encoding == genre_encoding

        with open(output_dir / "feature_statistics.json", "r") as f:
            stats = json.load(f)
            assert stats["n_movies"] == len(movie_features)
            assert stats["n_users"] == len(user_features)

    def test_process_full_pipeline(self, sample_data_path, tmp_path):
        """Test full processing pipeline."""
        fe = FeatureEngineering(sample_data_path)
        output_dir = tmp_path / "features"

        fe.process(output_dir)

        # Check all output files
        assert (output_dir / "movie_features.parquet").exists()
        assert (output_dir / "user_features.parquet").exists()
        assert (output_dir / "genre_encoding.json").exists()
        assert (output_dir / "feature_statistics.json").exists()

        # Verify loaded data
        movie_features = pd.read_parquet(output_dir / "movie_features.parquet")
        user_features = pd.read_parquet(output_dir / "user_features.parquet")

        assert len(movie_features) == 3
        assert len(user_features) == 3


class TestFeatureEngineeringEdgeCases:
    """Test edge cases for feature engineering."""

    def test_single_rating_per_item(self, tmp_path, sample_movies):
        """Test when items have single ratings."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # One rating per movie
        ratings = pd.DataFrame(
            {
                "userId": [1, 2, 3],
                "movieId": [1, 2, 3],
                "rating": [4.0, 5.0, 3.0],
                "timestamp": [1000, 2000, 3000],
            }
        )

        ratings.to_parquet(processed_dir / "train_ratings.parquet")
        ratings.to_parquet(processed_dir / "test_ratings.parquet")
        sample_movies.to_parquet(processed_dir / "movies.parquet")

        fe = FeatureEngineering(processed_dir)
        fe.load_data()
        movie_features = fe.engineer_movie_features()

        # rating_std should be 0 for single ratings
        assert (movie_features["rating_std"] == 0).all()

    def test_user_with_same_timestamp(self, tmp_path, sample_movies):
        """Test when user ratings have same timestamp."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Same timestamp for all ratings
        ratings = pd.DataFrame(
            {
                "userId": [1, 1, 1],
                "movieId": [1, 2, 3],
                "rating": [4.0, 5.0, 3.0],
                "timestamp": [1000, 1000, 1000],
            }
        )

        ratings.to_parquet(processed_dir / "train_ratings.parquet")
        ratings.to_parquet(processed_dir / "test_ratings.parquet")
        sample_movies.to_parquet(processed_dir / "movies.parquet")

        fe = FeatureEngineering(processed_dir)
        fe.load_data()
        user_features = fe.engineer_user_features()

        # activity_span_days should be 0
        assert user_features.loc[0, "activity_span_days"] == 0

    def test_movie_without_genres(self, tmp_path):
        """Test movie with no genres listed."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        ratings = pd.DataFrame(
            {
                "userId": [1],
                "movieId": [1],
                "rating": [4.0],
                "timestamp": [1000],
            }
        )

        movies = pd.DataFrame(
            {
                "movieId": [1],
                "title": ["Movie A (2000)"],
                "genres": ["(no genres listed)"],
            }
        )

        ratings.to_parquet(processed_dir / "train_ratings.parquet")
        ratings.to_parquet(processed_dir / "test_ratings.parquet")
        movies.to_parquet(processed_dir / "movies.parquet")

        fe = FeatureEngineering(processed_dir)
        fe.load_data()
        movie_features = fe.engineer_movie_features()

        # Should handle gracefully
        assert movie_features.loc[0, "n_genres"] == 0


class TestFeatureEngineeringDataTypes:
    """Test data type handling."""

    def test_year_extraction_handles_non_standard_formats(self, sample_data_path):
        """Test year extraction with various title formats."""
        fe = FeatureEngineering(sample_data_path)

        # Edge cases
        assert fe.extract_year_from_title("") == 0
        assert fe.extract_year_from_title("Movie (199)") == 0  # 3 digits
        assert fe.extract_year_from_title("Movie (19999)") == 0  # 5 digits
        assert fe.extract_year_from_title("Movie") == 0

    def test_genre_parsing_handles_empty_strings(self, sample_data_path):
        """Test genre parsing with empty and None values."""
        fe = FeatureEngineering(sample_data_path)

        assert fe.parse_genres("") == []
        assert fe.parse_genres(pd.NA) == []
        assert fe.parse_genres(float("nan")) == []
