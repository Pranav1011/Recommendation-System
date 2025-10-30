"""Unit tests for MovieLens data processor."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.processor import MovieLensProcessor


class TestMovieLensProcessor:
    """Tests for MovieLens processor class."""

    @pytest.fixture
    def sample_ratings_csv(self, tmp_path):
        """Create sample ratings.csv file."""
        ratings_data = """userId,movieId,rating,timestamp
1,1,4.0,964982703
1,2,3.5,964981247
2,1,5.0,964982224
2,3,4.5,964983815
3,2,3.0,964984134
3,3,4.0,964984474
1,3,2.5,964985128
"""
        ratings_file = tmp_path / "ratings.csv"
        ratings_file.write_text(ratings_data)
        return ratings_file

    @pytest.fixture
    def sample_movies_csv(self, tmp_path):
        """Create sample movies.csv file."""
        movies_data = """movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
"""
        movies_file = tmp_path / "movies.csv"
        movies_file.write_text(movies_data)
        return movies_file

    @pytest.fixture
    def sample_links_csv(self, tmp_path):
        """Create sample links.csv file."""
        links_data = """movieId,imdbId,tmdbId
1,114709,862
2,113497,8844
3,113228,15602
"""
        links_file = tmp_path / "links.csv"
        links_file.write_text(links_data)
        return links_file

    @pytest.fixture
    def sample_tags_csv(self, tmp_path):
        """Create sample tags.csv file."""
        tags_data = """userId,movieId,tag,timestamp
1,1,pixar,1137207627
2,2,adventure,1137207627
"""
        tags_file = tmp_path / "tags.csv"
        tags_file.write_text(tags_data)
        return tags_file

    @pytest.fixture
    def sample_dataset(self, sample_ratings_csv, sample_movies_csv):
        """Create complete sample dataset."""
        data_path = sample_ratings_csv.parent
        return data_path

    def test_init(self, tmp_path):
        """Test processor initialization."""
        processor = MovieLensProcessor(tmp_path, "latest-small")
        assert processor.data_path == tmp_path
        assert processor.dataset_size == "latest-small"
        assert processor.file_format == "csv"
        assert processor.separator == ","

    def test_init_1m_format(self, tmp_path):
        """Test processor initialization for 1M dataset."""
        processor = MovieLensProcessor(tmp_path, "1m")
        assert processor.file_format == "dat"
        assert processor.separator == "::"

    def test_load_data_csv(
        self, sample_ratings_csv, sample_movies_csv, sample_links_csv, sample_tags_csv
    ):
        """Test loading CSV format data."""
        data_path = sample_ratings_csv.parent
        processor = MovieLensProcessor(data_path, "latest-small")
        processor.load_data()

        assert processor.ratings_df is not None
        assert len(processor.ratings_df) == 7
        assert processor.movies_df is not None
        assert len(processor.movies_df) == 3
        assert processor.links_df is not None
        assert processor.tags_df is not None

    def test_load_data_missing_ratings(self, tmp_path):
        """Test that missing ratings file raises error."""
        processor = MovieLensProcessor(tmp_path, "latest-small")
        with pytest.raises(FileNotFoundError, match="Ratings file not found"):
            processor.load_data()

    def test_load_data_missing_movies(self, sample_ratings_csv, tmp_path):
        """Test that missing movies file raises error."""
        processor = MovieLensProcessor(sample_ratings_csv.parent, "latest-small")
        with pytest.raises(FileNotFoundError, match="Movies file not found"):
            processor.load_data()

    def test_validate_data_success(self, sample_dataset):
        """Test data validation with clean data."""
        processor = MovieLensProcessor(sample_dataset, "latest-small")
        processor.load_data()
        processor.validate_data()  # Should not raise

        assert processor.ratings_df is not None
        assert len(processor.ratings_df) == 7

    def test_validate_data_invalid_ratings(self, tmp_path):
        """Test validation catches invalid rating values."""
        # Create ratings with invalid values
        ratings_data = """userId,movieId,rating,timestamp
1,1,6.0,964982703
2,2,3.5,964981247
"""
        ratings_file = tmp_path / "ratings.csv"
        ratings_file.write_text(ratings_data)

        movies_data = """movieId,title,genres
1,Movie 1,Action
2,Movie 2,Comedy
"""
        movies_file = tmp_path / "movies.csv"
        movies_file.write_text(movies_data)

        processor = MovieLensProcessor(tmp_path, "latest-small")
        processor.load_data()

        with pytest.raises(ValueError, match="Invalid rating range"):
            processor.validate_data()

    def test_validate_data_removes_duplicates(self, tmp_path):
        """Test that duplicate ratings are removed."""
        ratings_data = """userId,movieId,rating,timestamp
1,1,4.0,964982703
1,1,5.0,964982704
2,2,3.5,964981247
"""
        ratings_file = tmp_path / "ratings.csv"
        ratings_file.write_text(ratings_data)

        movies_data = """movieId,title,genres
1,Movie 1,Action
2,Movie 2,Comedy
"""
        movies_file = tmp_path / "movies.csv"
        movies_file.write_text(movies_data)

        processor = MovieLensProcessor(tmp_path, "latest-small")
        processor.load_data()

        assert len(processor.ratings_df) == 3
        processor.validate_data()
        assert len(processor.ratings_df) == 2  # Duplicate removed

    def test_validate_data_filters_missing_movies(self, tmp_path):
        """Test that ratings for missing movies are filtered out."""
        ratings_data = """userId,movieId,rating,timestamp
1,1,4.0,964982703
1,2,3.5,964981247
2,999,5.0,964982224
"""
        ratings_file = tmp_path / "ratings.csv"
        ratings_file.write_text(ratings_data)

        movies_data = """movieId,title,genres
1,Movie 1,Action
2,Movie 2,Comedy
"""
        movies_file = tmp_path / "movies.csv"
        movies_file.write_text(movies_data)

        processor = MovieLensProcessor(tmp_path, "latest-small")
        processor.load_data()

        assert len(processor.ratings_df) == 3
        processor.validate_data()
        assert len(processor.ratings_df) == 2  # Rating for movie 999 removed

    def test_create_train_test_split_temporal(self, sample_dataset):
        """Test temporal train/test split."""
        processor = MovieLensProcessor(sample_dataset, "latest-small")
        processor.load_data()
        processor.validate_data()

        train_df, test_df = processor.create_train_test_split(
            test_size=0.3, temporal=True
        )

        assert len(train_df) + len(test_df) == len(processor.ratings_df)
        assert len(test_df) == pytest.approx(len(processor.ratings_df) * 0.3, abs=1)

        # Verify temporal ordering (train timestamps < test timestamps)
        max_train_timestamp = train_df["timestamp"].max()
        min_test_timestamp = test_df["timestamp"].min()
        assert max_train_timestamp <= min_test_timestamp

    def test_create_train_test_split_random(self, sample_dataset):
        """Test random train/test split."""
        processor = MovieLensProcessor(sample_dataset, "latest-small")
        processor.load_data()
        processor.validate_data()

        train_df, test_df = processor.create_train_test_split(
            test_size=0.3, temporal=False
        )

        assert len(train_df) + len(test_df) == len(processor.ratings_df)
        assert len(test_df) == pytest.approx(len(processor.ratings_df) * 0.3, abs=1)

    def test_compute_statistics(self, sample_dataset):
        """Test statistics computation."""
        processor = MovieLensProcessor(sample_dataset, "latest-small")
        processor.load_data()
        processor.validate_data()

        train_df, _ = processor.create_train_test_split(test_size=0.2)
        stats = processor.compute_statistics(train_df)

        assert "n_users" in stats
        assert "n_movies" in stats
        assert "n_ratings" in stats
        assert "sparsity" in stats
        assert "avg_rating" in stats
        assert "rating_std" in stats
        assert "ratings_per_user" in stats
        assert "ratings_per_movie" in stats

        assert stats["n_users"] > 0
        assert stats["n_movies"] > 0
        assert stats["n_ratings"] == len(train_df)
        assert 0 <= stats["sparsity"] <= 1
        assert 0 <= stats["avg_rating"] <= 5

    def test_save_to_parquet(self, sample_dataset, tmp_path):
        """Test saving data to Parquet files."""
        processor = MovieLensProcessor(sample_dataset, "latest-small")
        processor.load_data()
        processor.validate_data()

        train_df, test_df = processor.create_train_test_split(test_size=0.2)
        stats = processor.compute_statistics(train_df)

        output_dir = tmp_path / "output"
        processor.save_to_parquet(train_df, test_df, output_dir, stats)

        # Check files exist
        assert (output_dir / "train_ratings.parquet").exists()
        assert (output_dir / "test_ratings.parquet").exists()
        assert (output_dir / "movies.parquet").exists()
        assert (output_dir / "statistics.json").exists()

        # Verify we can read the files back
        loaded_train = pd.read_parquet(output_dir / "train_ratings.parquet")
        assert len(loaded_train) == len(train_df)

        loaded_test = pd.read_parquet(output_dir / "test_ratings.parquet")
        assert len(loaded_test) == len(test_df)

        loaded_movies = pd.read_parquet(output_dir / "movies.parquet")
        assert len(loaded_movies) == len(processor.movies_df)

        # Verify statistics JSON
        with open(output_dir / "statistics.json") as f:
            loaded_stats = json.load(f)
        assert loaded_stats["n_ratings"] == stats["n_ratings"]

    def test_process_full_pipeline(self, sample_dataset, tmp_path):
        """Test full processing pipeline."""
        processor = MovieLensProcessor(sample_dataset, "latest-small")
        output_dir = tmp_path / "processed"

        stats = processor.process(output_dir, test_size=0.2, temporal=True)

        # Verify outputs exist
        assert (output_dir / "train_ratings.parquet").exists()
        assert (output_dir / "test_ratings.parquet").exists()
        assert (output_dir / "movies.parquet").exists()
        assert (output_dir / "statistics.json").exists()

        # Verify statistics
        assert stats["n_users"] > 0
        assert stats["n_movies"] > 0
        assert stats["n_ratings"] > 0

    def test_process_creates_output_dir(self, sample_dataset, tmp_path):
        """Test that process creates output directory if it doesn't exist."""
        processor = MovieLensProcessor(sample_dataset, "latest-small")
        output_dir = tmp_path / "new_dir" / "processed"

        assert not output_dir.exists()
        processor.process(output_dir, test_size=0.2)
        assert output_dir.exists()

    def test_datatypes_are_optimized(self, sample_dataset):
        """Test that data types are optimized for memory."""
        processor = MovieLensProcessor(sample_dataset, "latest-small")
        processor.load_data()

        # Check ratings dtypes
        assert processor.ratings_df["userId"].dtype == "int32"
        assert processor.ratings_df["movieId"].dtype == "int32"
        assert processor.ratings_df["rating"].dtype == "float32"


class TestProcessorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataset(self, tmp_path):
        """Test handling of empty ratings file."""
        ratings_data = "userId,movieId,rating,timestamp\n"
        (tmp_path / "ratings.csv").write_text(ratings_data)

        movies_data = "movieId,title,genres\n"
        (tmp_path / "movies.csv").write_text(movies_data)

        processor = MovieLensProcessor(tmp_path, "latest-small")
        processor.load_data()

        assert len(processor.ratings_df) == 0
        assert len(processor.movies_df) == 0

    def test_single_user_dataset(self, tmp_path):
        """Test processing with single user."""
        ratings_data = """userId,movieId,rating,timestamp
1,1,4.0,964982703
1,2,3.5,964981247
1,3,5.0,964982224
"""
        (tmp_path / "ratings.csv").write_text(ratings_data)

        movies_data = """movieId,title,genres
1,Movie 1,Action
2,Movie 2,Comedy
3,Movie 3,Drama
"""
        (tmp_path / "movies.csv").write_text(movies_data)

        processor = MovieLensProcessor(tmp_path, "latest-small")
        processor.load_data()
        processor.validate_data()

        train_df, test_df = processor.create_train_test_split(test_size=0.2)
        stats = processor.compute_statistics(train_df)

        assert stats["n_users"] == 1

    def test_all_same_ratings(self, tmp_path):
        """Test dataset with all same rating values."""
        ratings_data = """userId,movieId,rating,timestamp
1,1,4.0,964982703
2,2,4.0,964981247
3,3,4.0,964982224
"""
        (tmp_path / "ratings.csv").write_text(ratings_data)

        movies_data = """movieId,title,genres
1,Movie 1,Action
2,Movie 2,Comedy
3,Movie 3,Drama
"""
        (tmp_path / "movies.csv").write_text(movies_data)

        processor = MovieLensProcessor(tmp_path, "latest-small")
        processor.load_data()
        processor.validate_data()

        train_df, _ = processor.create_train_test_split(test_size=0.2)
        stats = processor.compute_statistics(train_df)

        assert stats["avg_rating"] == 4.0
        assert stats["rating_std"] == 0.0
