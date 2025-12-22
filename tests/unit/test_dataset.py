"""
Unit tests for MovieLens Dataset.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.dataset import MovieLensDataset, create_dataloaders


@pytest.fixture
def mock_ratings_data():
    """Create mock ratings DataFrame."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 4, 4],
            "movieId": [10, 20, 10, 30, 20, 40, 30, 40],
            "rating": [4.0, 5.0, 3.0, 4.5, 5.0, 3.5, 4.0, 2.5],
            "timestamp": [100, 200, 300, 400, 500, 600, 700, 800],
        }
    )


@pytest.fixture
def mock_user_features_data():
    """Create mock user features DataFrame."""
    return pd.DataFrame(
        {
            "userId": [1, 2, 3, 4],
            "avg_rating": [4.5, 3.75, 4.25, 3.25],
            "rating_count": [2.0, 2.0, 2.0, 2.0],
            "genre_action": [0.5, 0.0, 0.5, 1.0],
        }
    )


@pytest.fixture
def mock_movie_features_data():
    """Create mock movie features DataFrame."""
    return pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "avg_rating": [3.5, 4.5, 4.25, 3.0],
            "rating_count": [2.0, 2.0, 2.0, 2.0],
            "release_year": [2000.0, 2010.0, 2015.0, 2020.0],
        }
    )


class TestMovieLensDataset:
    """Test MovieLensDataset class."""

    @patch("src.training.dataset.pd.read_parquet")
    def test_initialization_basic(self, mock_read_parquet, mock_ratings_data):
        """Test basic dataset initialization without features."""
        mock_read_parquet.return_value = mock_ratings_data

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=False,
        )

        assert len(dataset) == 8
        assert dataset.get_num_users() == 4
        assert dataset.get_num_movies() == 4
        mock_read_parquet.assert_called_once()

    @patch("src.training.dataset.pd.read_parquet")
    @patch.object(Path, "exists")
    def test_initialization_with_features(
        self,
        mock_exists,
        mock_read_parquet,
        mock_ratings_data,
        mock_user_features_data,
        mock_movie_features_data,
    ):
        """Test dataset initialization with user and movie features."""
        mock_exists.return_value = True

        # Mock read_parquet to return different data based on the file being read
        def read_parquet_side_effect(path):
            path_str = str(path)
            if "ratings" in path_str:
                return mock_ratings_data.copy()
            elif "user_features" in path_str:
                return mock_user_features_data.copy()
            elif "movie_features" in path_str:
                return mock_movie_features_data.copy()
            return pd.DataFrame()

        mock_read_parquet.side_effect = read_parquet_side_effect

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            user_features_path=Path("user_features.parquet"),
            movie_features_path=Path("movie_features.parquet"),
            negative_sampling=False,
        )

        user_dim, movie_dim = dataset.get_feature_dims()
        assert user_dim == 3  # avg_rating, rating_count, genre_action
        assert movie_dim == 3  # avg_rating, rating_count, release_year
        assert dataset.user_features.shape == (4, 3)
        assert dataset.movie_features.shape == (4, 3)

    @patch("src.training.dataset.pd.read_parquet")
    @patch.object(Path, "exists")
    def test_initialization_missing_features(
        self, mock_exists, mock_read_parquet, mock_ratings_data
    ):
        """Test dataset initialization when feature files don't exist."""
        mock_exists.return_value = False
        mock_read_parquet.return_value = mock_ratings_data

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            user_features_path=Path("user_features.parquet"),
            movie_features_path=Path("movie_features.parquet"),
            negative_sampling=False,
        )

        user_dim, movie_dim = dataset.get_feature_dims()
        assert user_dim == 0
        assert movie_dim == 0
        assert dataset.user_features is None
        assert dataset.movie_features is None

    @patch("src.training.dataset.pd.read_parquet")
    def test_user_movie_mappings(self, mock_read_parquet, mock_ratings_data):
        """Test user and movie ID to index mappings."""
        mock_read_parquet.return_value = mock_ratings_data

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=False,
        )

        # Check unique users are sorted
        assert dataset.unique_users == [1, 2, 3, 4]
        assert dataset.unique_movies == [10, 20, 30, 40]

        # Check mappings
        assert dataset.user_to_idx == {1: 0, 2: 1, 3: 2, 4: 3}
        assert dataset.movie_to_idx == {10: 0, 20: 1, 30: 2, 40: 3}

        # Check that ratings_df has index columns
        assert "user_idx" in dataset.ratings_df.columns
        assert "movie_idx" in dataset.ratings_df.columns

    @patch("src.training.dataset.pd.read_parquet")
    def test_len(self, mock_read_parquet, mock_ratings_data):
        """Test __len__ method."""
        mock_read_parquet.return_value = mock_ratings_data

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=False,
        )

        assert len(dataset) == 8

    @patch("src.training.dataset.pd.read_parquet")
    def test_getitem_without_features(self, mock_read_parquet, mock_ratings_data):
        """Test __getitem__ without features or negative sampling."""
        mock_read_parquet.return_value = mock_ratings_data

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=False,
        )

        sample = dataset[0]

        assert "user_id" in sample
        assert "movie_id" in sample
        assert "rating" in sample
        assert "user_features" not in sample
        assert "movie_features" not in sample
        assert "neg_movie_ids" not in sample

        assert isinstance(sample["user_id"], torch.Tensor)
        assert isinstance(sample["movie_id"], torch.Tensor)
        assert isinstance(sample["rating"], torch.Tensor)

        assert sample["user_id"].dtype == torch.long
        assert sample["movie_id"].dtype == torch.long
        assert sample["rating"].dtype == torch.float32

    @patch("src.training.dataset.pd.read_parquet")
    @patch.object(Path, "exists")
    def test_getitem_with_features(
        self,
        mock_exists,
        mock_read_parquet,
        mock_ratings_data,
        mock_user_features_data,
        mock_movie_features_data,
    ):
        """Test __getitem__ with user and movie features."""
        mock_exists.return_value = True

        def read_parquet_side_effect(path):
            path_str = str(path)
            if "ratings" in path_str:
                return mock_ratings_data.copy()
            elif "user_features" in path_str:
                return mock_user_features_data.copy()
            elif "movie_features" in path_str:
                return mock_movie_features_data.copy()
            return pd.DataFrame()

        mock_read_parquet.side_effect = read_parquet_side_effect

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            user_features_path=Path("user_features.parquet"),
            movie_features_path=Path("movie_features.parquet"),
            negative_sampling=False,
        )

        sample = dataset[0]

        assert "user_features" in sample
        assert "movie_features" in sample

        assert sample["user_features"].dtype == torch.float32
        assert sample["movie_features"].dtype == torch.float32
        assert sample["user_features"].shape == (3,)
        assert sample["movie_features"].shape == (3,)

    @patch("src.training.dataset.pd.read_parquet")
    def test_getitem_with_negative_sampling(self, mock_read_parquet, mock_ratings_data):
        """Test __getitem__ with random negative sampling."""
        mock_read_parquet.return_value = mock_ratings_data

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=True,
            n_negatives=2,
        )

        sample = dataset[0]

        assert "neg_movie_ids" in sample
        assert isinstance(sample["neg_movie_ids"], torch.Tensor)
        assert sample["neg_movie_ids"].dtype == torch.long
        assert sample["neg_movie_ids"].shape == (2,)

        # Negative samples should be different from positive sample
        movie_id = sample["movie_id"].item()
        neg_movie_ids = sample["neg_movie_ids"].tolist()
        assert movie_id not in neg_movie_ids

    @patch("src.training.dataset.pd.read_parquet")
    def test_getitem_with_hard_negative_sampler(
        self, mock_read_parquet, mock_ratings_data
    ):
        """Test __getitem__ with hard negative sampler."""
        mock_read_parquet.return_value = mock_ratings_data

        # Create mock hard negative sampler
        mock_sampler = Mock()
        mock_sampler.sample.return_value = np.array([2, 3])

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=True,
            n_negatives=2,
            hard_negative_sampler=mock_sampler,
        )

        sample = dataset[0]

        assert "neg_movie_ids" in sample
        assert sample["neg_movie_ids"].shape == (2,)

        # Check that hard negative sampler was called
        mock_sampler.sample.assert_called_once_with(1, n_negatives=2)

    @patch("src.training.dataset.pd.read_parquet")
    def test_negative_sampling_with_insufficient_candidates(self, mock_read_parquet):
        """Test negative sampling when there aren't enough negative candidates."""
        # Create a scenario where user has rated 3 out of 4 movies
        # User 1 has rated movies 10, 20, 30 but not 40
        # User 2 has rated only movie 40
        ratings_data = pd.DataFrame(
            {
                "userId": [1, 1, 1, 2],
                "movieId": [10, 20, 30, 40],
                "rating": [4.0, 5.0, 3.0, 4.0],
                "timestamp": [100, 200, 300, 400],
            }
        )
        mock_read_parquet.return_value = ratings_data

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=True,
            n_negatives=5,  # More than available negative candidates (only 1 unrated movie)
        )

        sample = dataset[0]  # User 1's first rating

        # Should still return n_negatives samples (with replacement since only 1 negative available)
        assert sample["neg_movie_ids"].shape == (5,)

    @patch("src.training.dataset.pd.read_parquet")
    def test_get_num_users(self, mock_read_parquet, mock_ratings_data):
        """Test get_num_users method."""
        mock_read_parquet.return_value = mock_ratings_data

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=False,
        )

        assert dataset.get_num_users() == 4

    @patch("src.training.dataset.pd.read_parquet")
    def test_get_num_movies(self, mock_read_parquet, mock_ratings_data):
        """Test get_num_movies method."""
        mock_read_parquet.return_value = mock_ratings_data

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=False,
        )

        assert dataset.get_num_movies() == 4

    @patch("src.training.dataset.pd.read_parquet")
    def test_get_feature_dims_no_features(self, mock_read_parquet, mock_ratings_data):
        """Test get_feature_dims when no features are loaded."""
        mock_read_parquet.return_value = mock_ratings_data

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=False,
        )

        user_dim, movie_dim = dataset.get_feature_dims()
        assert user_dim == 0
        assert movie_dim == 0

    @patch("src.training.dataset.pd.read_parquet")
    @patch.object(Path, "exists")
    def test_get_feature_dims_with_features(
        self,
        mock_exists,
        mock_read_parquet,
        mock_ratings_data,
        mock_user_features_data,
        mock_movie_features_data,
    ):
        """Test get_feature_dims when features are loaded."""
        mock_exists.return_value = True

        def read_parquet_side_effect(path):
            path_str = str(path)
            if "ratings" in path_str:
                return mock_ratings_data.copy()
            elif "user_features" in path_str:
                return mock_user_features_data.copy()
            elif "movie_features" in path_str:
                return mock_movie_features_data.copy()
            return pd.DataFrame()

        mock_read_parquet.side_effect = read_parquet_side_effect

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            user_features_path=Path("user_features.parquet"),
            movie_features_path=Path("movie_features.parquet"),
            negative_sampling=False,
        )

        user_dim, movie_dim = dataset.get_feature_dims()
        assert user_dim == 3
        assert movie_dim == 3

    @patch("src.training.dataset.pd.read_parquet")
    def test_user_positive_movies_dict(self, mock_read_parquet, mock_ratings_data):
        """Test that user_positive_movies dict is created correctly."""
        mock_read_parquet.return_value = mock_ratings_data

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=True,
            n_negatives=2,
        )

        # Check that user_positive_movies exists
        assert hasattr(dataset, "user_positive_movies")
        assert isinstance(dataset.user_positive_movies, dict)

        # User with index 0 (userId=1) has rated movies with indices 0 and 1
        assert 0 in dataset.user_positive_movies[0]  # movieId=10 -> idx=0
        assert 1 in dataset.user_positive_movies[0]  # movieId=20 -> idx=1

    @patch("src.training.dataset.pd.read_parquet")
    @patch.object(Path, "exists")
    def test_feature_loading_with_missing_users(
        self,
        mock_exists,
        mock_read_parquet,
        mock_ratings_data,
    ):
        """Test feature loading when some users are missing from feature file."""
        mock_exists.return_value = True

        # Create user features with only 2 users (missing users 3 and 4)
        partial_user_features = pd.DataFrame(
            {
                "userId": [1, 2],
                "avg_rating": [4.5, 3.75],
                "rating_count": [2.0, 2.0],
            }
        )

        def read_parquet_side_effect(path):
            path_str = str(path)
            if "ratings" in path_str:
                return mock_ratings_data.copy()
            elif "user_features" in path_str:
                return partial_user_features.copy()
            return pd.DataFrame()

        mock_read_parquet.side_effect = read_parquet_side_effect

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            user_features_path=Path("user_features.parquet"),
            negative_sampling=False,
        )

        # Features should be zero-initialized for missing users
        assert dataset.user_features.shape == (4, 2)  # 4 users, 2 features

        # Check that missing users have zero features
        assert np.allclose(dataset.user_features[2], 0.0)  # User 3 (idx 2)
        assert np.allclose(dataset.user_features[3], 0.0)  # User 4 (idx 3)


class TestCreateDataloaders:
    """Test create_dataloaders factory function."""

    @patch("src.training.dataset.torch.cuda.is_available")
    @patch("src.training.dataset.MovieLensDataset")
    def test_create_dataloaders_basic(
        self, mock_dataset_class, mock_cuda_available, mock_ratings_data
    ):
        """Test basic dataloader creation."""
        mock_cuda_available.return_value = False

        # Create mock dataset instances
        mock_train_dataset = Mock()
        mock_test_dataset = Mock()
        mock_train_dataset.__len__ = Mock(return_value=1000)
        mock_test_dataset.__len__ = Mock(return_value=200)

        mock_dataset_class.side_effect = [mock_train_dataset, mock_test_dataset]

        train_loader, test_loader = create_dataloaders(
            train_ratings_path=Path("train.parquet"),
            test_ratings_path=Path("test.parquet"),
            batch_size=128,
            num_workers=2,
            negative_sampling=False,
        )

        # Check that datasets were created with correct parameters
        assert mock_dataset_class.call_count == 2

        # Check train dataset call
        train_call_kwargs = mock_dataset_class.call_args_list[0][1]
        assert train_call_kwargs["ratings_path"] == Path("train.parquet")
        assert train_call_kwargs["negative_sampling"] is False

        # Check test dataset call
        test_call_kwargs = mock_dataset_class.call_args_list[1][1]
        assert test_call_kwargs["ratings_path"] == Path("test.parquet")
        assert test_call_kwargs["negative_sampling"] is False

        # Check that dataloaders were created
        assert train_loader is not None
        assert test_loader is not None

    @patch("src.training.dataset.torch.cuda.is_available")
    @patch("src.training.dataset.MovieLensDataset")
    def test_create_dataloaders_with_features(
        self, mock_dataset_class, mock_cuda_available
    ):
        """Test dataloader creation with feature paths."""
        mock_cuda_available.return_value = False

        mock_train_dataset = Mock()
        mock_test_dataset = Mock()
        mock_train_dataset.__len__ = Mock(return_value=1000)
        mock_test_dataset.__len__ = Mock(return_value=200)

        mock_dataset_class.side_effect = [mock_train_dataset, mock_test_dataset]

        user_features_path = Path("user_features.parquet")
        movie_features_path = Path("movie_features.parquet")

        train_loader, test_loader = create_dataloaders(
            train_ratings_path=Path("train.parquet"),
            test_ratings_path=Path("test.parquet"),
            user_features_path=user_features_path,
            movie_features_path=movie_features_path,
            batch_size=512,
            num_workers=4,
            negative_sampling=True,
            n_negatives=4,
        )

        # Check that datasets were created with feature paths
        train_call_kwargs = mock_dataset_class.call_args_list[0][1]
        assert train_call_kwargs["user_features_path"] == user_features_path
        assert train_call_kwargs["movie_features_path"] == movie_features_path
        assert train_call_kwargs["negative_sampling"] is True
        assert train_call_kwargs["n_negatives"] == 4

        # Test dataset should NOT have negative sampling
        test_call_kwargs = mock_dataset_class.call_args_list[1][1]
        assert test_call_kwargs["negative_sampling"] is False

    @patch("src.training.dataset.torch.cuda.is_available")
    @patch("src.training.dataset.MovieLensDataset")
    def test_create_dataloaders_with_hard_negative_sampler(
        self, mock_dataset_class, mock_cuda_available
    ):
        """Test dataloader creation with hard negative sampler."""
        mock_cuda_available.return_value = False

        mock_train_dataset = Mock()
        mock_test_dataset = Mock()
        mock_train_dataset.__len__ = Mock(return_value=1000)
        mock_test_dataset.__len__ = Mock(return_value=200)

        mock_dataset_class.side_effect = [mock_train_dataset, mock_test_dataset]

        mock_sampler = Mock()

        train_loader, test_loader = create_dataloaders(
            train_ratings_path=Path("train.parquet"),
            test_ratings_path=Path("test.parquet"),
            batch_size=256,
            num_workers=4,
            negative_sampling=True,
            hard_negative_sampler=mock_sampler,
        )

        # Check that hard negative sampler was passed to train dataset
        train_call_kwargs = mock_dataset_class.call_args_list[0][1]
        assert train_call_kwargs["hard_negative_sampler"] == mock_sampler

    @patch("src.training.dataset.torch.cuda.is_available")
    @patch("src.training.dataset.MovieLensDataset")
    def test_create_dataloaders_pin_memory_cuda(
        self, mock_dataset_class, mock_cuda_available
    ):
        """Test that pin_memory is enabled when CUDA is available."""
        mock_cuda_available.return_value = True

        mock_train_dataset = Mock()
        mock_test_dataset = Mock()
        mock_train_dataset.__len__ = Mock(return_value=1000)
        mock_test_dataset.__len__ = Mock(return_value=200)

        mock_dataset_class.side_effect = [mock_train_dataset, mock_test_dataset]

        with patch("src.training.dataset.torch.utils.data.DataLoader") as mock_loader:
            create_dataloaders(
                train_ratings_path=Path("train.parquet"),
                test_ratings_path=Path("test.parquet"),
                batch_size=128,
                num_workers=2,
            )

            # Check that pin_memory=True was passed
            train_call_kwargs = mock_loader.call_args_list[0][1]
            assert train_call_kwargs["pin_memory"] is True

    @patch("src.training.dataset.torch.cuda.is_available")
    @patch("src.training.dataset.MovieLensDataset")
    def test_create_dataloaders_no_pin_memory_cpu(
        self, mock_dataset_class, mock_cuda_available
    ):
        """Test that pin_memory is disabled when CUDA is not available."""
        mock_cuda_available.return_value = False

        mock_train_dataset = Mock()
        mock_test_dataset = Mock()
        mock_train_dataset.__len__ = Mock(return_value=1000)
        mock_test_dataset.__len__ = Mock(return_value=200)

        mock_dataset_class.side_effect = [mock_train_dataset, mock_test_dataset]

        with patch("src.training.dataset.torch.utils.data.DataLoader") as mock_loader:
            create_dataloaders(
                train_ratings_path=Path("train.parquet"),
                test_ratings_path=Path("test.parquet"),
                batch_size=128,
                num_workers=2,
            )

            # Check that pin_memory=False was passed
            train_call_kwargs = mock_loader.call_args_list[0][1]
            assert train_call_kwargs["pin_memory"] is False

    @patch("src.training.dataset.torch.cuda.is_available")
    @patch("src.training.dataset.MovieLensDataset")
    def test_create_dataloaders_shuffle_settings(
        self, mock_dataset_class, mock_cuda_available
    ):
        """Test that train loader shuffles but test loader doesn't."""
        mock_cuda_available.return_value = False

        mock_train_dataset = Mock()
        mock_test_dataset = Mock()
        mock_train_dataset.__len__ = Mock(return_value=1000)
        mock_test_dataset.__len__ = Mock(return_value=200)

        mock_dataset_class.side_effect = [mock_train_dataset, mock_test_dataset]

        with patch("src.training.dataset.torch.utils.data.DataLoader") as mock_loader:
            create_dataloaders(
                train_ratings_path=Path("train.parquet"),
                test_ratings_path=Path("test.parquet"),
                batch_size=128,
                num_workers=2,
            )

            # Check train loader shuffle=True
            train_call_kwargs = mock_loader.call_args_list[0][1]
            assert train_call_kwargs["shuffle"] is True

            # Check test loader shuffle=False
            test_call_kwargs = mock_loader.call_args_list[1][1]
            assert test_call_kwargs["shuffle"] is False


class TestDatasetEdgeCases:
    """Test edge cases and error conditions."""

    @patch("src.training.dataset.pd.read_parquet")
    def test_empty_ratings_dataframe(self, mock_read_parquet):
        """Test handling of empty ratings DataFrame."""
        empty_df = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
        mock_read_parquet.return_value = empty_df

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=False,
        )

        assert len(dataset) == 0
        assert dataset.get_num_users() == 0
        assert dataset.get_num_movies() == 0

    @patch("src.training.dataset.pd.read_parquet")
    def test_single_rating(self, mock_read_parquet):
        """Test dataset with single rating."""
        single_rating = pd.DataFrame(
            {
                "userId": [1],
                "movieId": [10],
                "rating": [4.0],
                "timestamp": [100],
            }
        )
        mock_read_parquet.return_value = single_rating

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            negative_sampling=False,
        )

        assert len(dataset) == 1
        assert dataset.get_num_users() == 1
        assert dataset.get_num_movies() == 1

        sample = dataset[0]
        assert sample["rating"].item() == 4.0

    @patch("src.training.dataset.pd.read_parquet")
    @patch.object(Path, "exists")
    def test_features_with_non_numeric_columns(
        self, mock_exists, mock_read_parquet, mock_ratings_data
    ):
        """Test that non-numeric feature columns are filtered out."""
        mock_exists.return_value = True

        # Create features with non-numeric columns
        mixed_features = pd.DataFrame(
            {
                "userId": [1, 2, 3, 4],
                "avg_rating": [4.5, 3.75, 4.25, 3.25],
                "username": ["user1", "user2", "user3", "user4"],  # Non-numeric
                "genre": ["Action", "Drama", "Comedy", "Thriller"],  # Non-numeric
            }
        )

        def read_parquet_side_effect(path):
            path_str = str(path)
            if "ratings" in path_str:
                return mock_ratings_data.copy()
            elif "user_features" in path_str:
                return mixed_features.copy()
            return pd.DataFrame()

        mock_read_parquet.side_effect = read_parquet_side_effect

        dataset = MovieLensDataset(
            ratings_path=Path("dummy_ratings.parquet"),
            user_features_path=Path("user_features.parquet"),
            negative_sampling=False,
        )

        # Should only have 1 numeric feature (avg_rating)
        user_dim, _ = dataset.get_feature_dims()
        assert user_dim == 1

    @patch("src.training.dataset.pd.read_parquet")
    def test_different_n_negatives_values(self, mock_read_parquet, mock_ratings_data):
        """Test negative sampling with different n_negatives values."""
        mock_read_parquet.return_value = mock_ratings_data

        for n_negatives in [1, 2, 4, 8]:
            dataset = MovieLensDataset(
                ratings_path=Path("dummy_ratings.parquet"),
                negative_sampling=True,
                n_negatives=n_negatives,
            )

            sample = dataset[0]
            assert sample["neg_movie_ids"].shape == (n_negatives,)
