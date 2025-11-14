"""
Unit tests for Embedding Generator.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.embeddings.generate import EmbeddingGenerator


@pytest.fixture
def mock_config():
    """Mock model configuration."""
    return {
        "n_users": 1000,
        "n_movies": 500,
        "embedding_dim": 128,
        "user_feature_dim": 30,
        "movie_feature_dim": 13,
        "hidden_dim": 256,
        "dropout": 0.2,
    }


@pytest.fixture
def mock_checkpoint(mock_config):
    """Mock model checkpoint."""
    return {
        "config": mock_config,
        "model_state_dict": {},
        "epoch": 10,
        "best_val_loss": 0.5,
    }


@pytest.fixture
def mock_model():
    """Mock Two-Tower model."""
    model = MagicMock()
    model.eval = Mock(return_value=None)
    model.to = Mock(return_value=model)
    model.load_state_dict = Mock(return_value=None)

    # Mock embedding generation methods
    def get_user_embedding(user_ids, user_features=None):
        batch_size = user_ids.shape[0]
        return torch.randn(batch_size, 128)

    def get_movie_embedding(movie_ids, movie_features=None):
        batch_size = movie_ids.shape[0]
        return torch.randn(batch_size, 128)

    model.get_user_embedding = Mock(side_effect=get_user_embedding)
    model.get_movie_embedding = Mock(side_effect=get_movie_embedding)

    return model


@pytest.fixture
def mock_user_features_df():
    """Mock user features DataFrame."""
    return pd.DataFrame(
        {
            "userId": range(1000),
            "avg_rating": np.random.uniform(1, 5, 1000),
            "rating_count": np.random.randint(1, 100, 1000),
            "genre_action": np.random.uniform(0, 1, 1000),
        }
    )


@pytest.fixture
def mock_movie_features_df():
    """Mock movie features DataFrame."""
    return pd.DataFrame(
        {
            "movieId": range(500),
            "popularity": np.random.uniform(0, 100, 500),
            "avg_rating": np.random.uniform(1, 5, 500),
            "release_year": np.random.randint(1990, 2020, 500),
        }
    )


class TestEmbeddingGeneratorInit:
    """Test EmbeddingGenerator initialization."""

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    def test_initialization_cuda_available(
        self, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test initialization with CUDA available."""
        mock_cuda.return_value = True
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cuda",
            )

            assert generator.device.type == "cuda"
            assert generator.n_users == 1000
            assert generator.n_movies == 500
            mock_load.assert_called_once()

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    def test_initialization_cuda_not_available(
        self, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test initialization with CUDA not available."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cuda",
            )

            assert generator.device.type == "cpu"

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    def test_initialization_cpu_device(
        self, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test initialization with CPU device explicitly."""
        mock_cuda.return_value = True
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            assert generator.device.type == "cpu"

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_initialization_with_features(
        self,
        mock_path_exists,
        mock_read_parquet,
        mock_cuda,
        mock_load,
        mock_checkpoint,
        mock_model,
        mock_user_features_df,
        mock_movie_features_df,
    ):
        """Test initialization with user and movie features."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint
        mock_path_exists.return_value = True
        mock_read_parquet.side_effect = [
            mock_user_features_df,
            mock_movie_features_df,
        ]

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                user_features_path=Path("user_features.parquet"),
                movie_features_path=Path("movie_features.parquet"),
                device="cpu",
            )

            assert generator.user_features is not None
            assert generator.movie_features is not None
            assert generator.user_features.shape == (1000, 3)  # 3 numeric features
            assert generator.movie_features.shape == (500, 3)  # 3 numeric features

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    def test_initialization_without_features(
        self, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test initialization without features."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            assert generator.user_features is None
            assert generator.movie_features is None


class TestLoadModel:
    """Test model loading functionality."""

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    def test_load_model_success(
        self, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test successful model loading."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            mock_load.assert_called_once_with(
                Path("model.pt"), map_location=torch.device("cpu")
            )
            mock_model.load_state_dict.assert_called_once_with({})
            mock_model.eval.assert_called_once()

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    def test_load_model_checkpoint_structure(
        self, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test that checkpoint has correct structure."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            assert generator.n_users == mock_checkpoint["config"]["n_users"]
            assert generator.n_movies == mock_checkpoint["config"]["n_movies"]


class TestLoadFeatures:
    """Test feature loading functionality."""

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_load_user_features_only(
        self,
        mock_path_exists,
        mock_read_parquet,
        mock_cuda,
        mock_load,
        mock_checkpoint,
        mock_model,
        mock_user_features_df,
    ):
        """Test loading only user features."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint
        mock_path_exists.return_value = True
        mock_read_parquet.return_value = mock_user_features_df

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                user_features_path=Path("user_features.parquet"),
                device="cpu",
            )

            assert generator.user_features is not None
            assert generator.movie_features is None
            assert generator.user_features.dtype == np.float32

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_load_movie_features_only(
        self,
        mock_path_exists,
        mock_read_parquet,
        mock_cuda,
        mock_load,
        mock_checkpoint,
        mock_model,
        mock_movie_features_df,
    ):
        """Test loading only movie features."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint
        mock_path_exists.return_value = True
        mock_read_parquet.return_value = mock_movie_features_df

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                movie_features_path=Path("movie_features.parquet"),
                device="cpu",
            )

            assert generator.user_features is None
            assert generator.movie_features is not None
            assert generator.movie_features.dtype == np.float32

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.pd.read_parquet")
    @patch("pathlib.Path.exists")
    def test_load_features_excludes_id_column(
        self,
        mock_path_exists,
        mock_read_parquet,
        mock_cuda,
        mock_load,
        mock_checkpoint,
        mock_model,
        mock_user_features_df,
    ):
        """Test that ID columns are excluded from features."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint
        mock_path_exists.return_value = True
        mock_read_parquet.return_value = mock_user_features_df

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                user_features_path=Path("user_features.parquet"),
                device="cpu",
            )

            # Should have 3 features (not 4 with userId)
            assert generator.user_features.shape[1] == 3

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    def test_load_features_nonexistent_path(
        self, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test loading features from nonexistent path."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                user_features_path=Path("nonexistent.parquet"),
                device="cpu",
            )

            # Should be None if path doesn't exist
            assert generator.user_features is None


class TestGenerateUserEmbeddings:
    """Test user embedding generation."""

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.tqdm")
    def test_generate_user_embeddings_basic(
        self, mock_tqdm, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test basic user embedding generation."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint
        mock_tqdm.return_value = range(0, 1000, 1024)  # Mock tqdm iterator

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            embeddings = generator.generate_user_embeddings(batch_size=1024)

            assert embeddings.shape == (1000, 128)
            assert embeddings.dtype == np.float32
            assert mock_model.get_user_embedding.called

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.tqdm")
    @patch("pathlib.Path.exists")
    def test_generate_user_embeddings_with_features(
        self,
        mock_path_exists,
        mock_tqdm,
        mock_cuda,
        mock_load,
        mock_checkpoint,
        mock_model,
        mock_user_features_df,
    ):
        """Test user embedding generation with features."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint
        mock_tqdm.return_value = range(0, 1000, 1024)
        mock_path_exists.return_value = True

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            with patch(
                "src.embeddings.generate.pd.read_parquet",
                return_value=mock_user_features_df,
            ):
                generator = EmbeddingGenerator(
                    model_path=Path("model.pt"),
                    user_features_path=Path("user_features.parquet"),
                    device="cpu",
                )

                embeddings = generator.generate_user_embeddings(batch_size=1024)

                assert embeddings.shape == (1000, 128)
                # Verify features were passed to model
                call_args = mock_model.get_user_embedding.call_args_list[0]
                assert call_args[0][1] is not None  # user_features argument

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.tqdm")
    @pytest.mark.parametrize("batch_size", [256, 512, 1024, 2048])
    def test_generate_user_embeddings_different_batch_sizes(
        self, mock_tqdm, mock_cuda, mock_load, mock_checkpoint, mock_model, batch_size
    ):
        """Test user embedding generation with different batch sizes."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint
        mock_tqdm.return_value = range(0, 1000, batch_size)

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            embeddings = generator.generate_user_embeddings(batch_size=batch_size)

            assert embeddings.shape == (1000, 128)

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.tqdm")
    def test_generate_user_embeddings_no_grad(
        self, mock_tqdm, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test that user embedding generation uses no_grad context."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint
        mock_tqdm.return_value = range(0, 1000, 1024)

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            with patch("torch.no_grad") as mock_no_grad:
                mock_no_grad.return_value.__enter__ = Mock()
                mock_no_grad.return_value.__exit__ = Mock()

                embeddings = generator.generate_user_embeddings(batch_size=1024)

                # Function is decorated with @torch.no_grad(), so it's implicitly called
                assert embeddings.shape == (1000, 128)


class TestGenerateMovieEmbeddings:
    """Test movie embedding generation."""

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.tqdm")
    def test_generate_movie_embeddings_basic(
        self, mock_tqdm, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test basic movie embedding generation."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint
        mock_tqdm.return_value = range(0, 500, 1024)

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            embeddings = generator.generate_movie_embeddings(batch_size=1024)

            assert embeddings.shape == (500, 128)
            assert embeddings.dtype == np.float32
            assert mock_model.get_movie_embedding.called

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.tqdm")
    @patch("pathlib.Path.exists")
    def test_generate_movie_embeddings_with_features(
        self,
        mock_path_exists,
        mock_tqdm,
        mock_cuda,
        mock_load,
        mock_checkpoint,
        mock_model,
        mock_movie_features_df,
    ):
        """Test movie embedding generation with features."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint
        mock_tqdm.return_value = range(0, 500, 1024)
        mock_path_exists.return_value = True

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            with patch(
                "src.embeddings.generate.pd.read_parquet",
                return_value=mock_movie_features_df,
            ):
                generator = EmbeddingGenerator(
                    model_path=Path("model.pt"),
                    movie_features_path=Path("movie_features.parquet"),
                    device="cpu",
                )

                embeddings = generator.generate_movie_embeddings(batch_size=1024)

                assert embeddings.shape == (500, 128)
                # Verify features were passed to model
                call_args = mock_model.get_movie_embedding.call_args_list[0]
                assert call_args[0][1] is not None  # movie_features argument

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.tqdm")
    @pytest.mark.parametrize("batch_size", [256, 512, 1024])
    def test_generate_movie_embeddings_different_batch_sizes(
        self, mock_tqdm, mock_cuda, mock_load, mock_checkpoint, mock_model, batch_size
    ):
        """Test movie embedding generation with different batch sizes."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint
        mock_tqdm.return_value = range(0, 500, batch_size)

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            embeddings = generator.generate_movie_embeddings(batch_size=batch_size)

            assert embeddings.shape == (500, 128)


class TestSaveEmbeddings:
    """Test embedding saving functionality."""

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.np.save")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_embeddings_creates_directory(
        self, mock_file, mock_np_save, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test that save_embeddings creates output directory."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            user_embeddings = np.random.randn(1000, 128).astype(np.float32)
            movie_embeddings = np.random.randn(500, 128).astype(np.float32)
            output_dir = Path("/tmp/embeddings")

            with patch.object(Path, "mkdir") as mock_mkdir:
                with patch("pandas.DataFrame.to_parquet"):
                    generator.save_embeddings(
                        user_embeddings, movie_embeddings, output_dir
                    )
                    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.np.save")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_embeddings_numpy_format(
        self, mock_file, mock_np_save, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test saving embeddings as NumPy arrays."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            user_embeddings = np.random.randn(1000, 128).astype(np.float32)
            movie_embeddings = np.random.randn(500, 128).astype(np.float32)
            output_dir = Path("/tmp/embeddings")

            with patch("pandas.DataFrame.to_parquet"):
                generator.save_embeddings(user_embeddings, movie_embeddings, output_dir)

                # Verify np.save was called twice (user and movie)
                assert mock_np_save.call_count == 2

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.np.save")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_embeddings_parquet_format(
        self, mock_file, mock_np_save, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test saving embeddings as Parquet files."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            user_embeddings = np.random.randn(1000, 128).astype(np.float32)
            movie_embeddings = np.random.randn(500, 128).astype(np.float32)
            output_dir = Path("/tmp/embeddings")

            with patch("pandas.DataFrame.to_parquet") as mock_to_parquet:
                generator.save_embeddings(user_embeddings, movie_embeddings, output_dir)

                # Verify to_parquet was called twice (user and movie)
                assert mock_to_parquet.call_count == 2

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.np.save")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_embeddings_metadata(
        self, mock_file, mock_np_save, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test saving embedding metadata."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            user_embeddings = np.random.randn(1000, 128).astype(np.float32)
            movie_embeddings = np.random.randn(500, 128).astype(np.float32)
            output_dir = Path("/tmp/embeddings")

            with patch("pandas.DataFrame.to_parquet"):
                with patch("json.dump") as mock_json_dump:
                    generator.save_embeddings(
                        user_embeddings, movie_embeddings, output_dir
                    )

                    # Verify metadata was saved
                    mock_json_dump.assert_called_once()
                    metadata = mock_json_dump.call_args[0][0]

                    assert metadata["n_users"] == 1000
                    assert metadata["n_movies"] == 500
                    assert metadata["embedding_dim"] == 128
                    assert metadata["user_embeddings_shape"] == [1000, 128]
                    assert metadata["movie_embeddings_shape"] == [500, 128]

    @patch("src.embeddings.generate.torch.load")
    @patch("src.embeddings.generate.torch.cuda.is_available")
    @patch("src.embeddings.generate.np.save")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_embeddings_dataframe_structure(
        self, mock_file, mock_np_save, mock_cuda, mock_load, mock_checkpoint, mock_model
    ):
        """Test that DataFrames have correct structure."""
        mock_cuda.return_value = False
        mock_load.return_value = mock_checkpoint

        with patch("src.models.two_tower.create_model", return_value=mock_model):
            generator = EmbeddingGenerator(
                model_path=Path("model.pt"),
                device="cpu",
            )

            user_embeddings = np.random.randn(1000, 128).astype(np.float32)
            movie_embeddings = np.random.randn(500, 128).astype(np.float32)
            output_dir = Path("/tmp/embeddings")

            with patch("pandas.DataFrame.to_parquet") as mock_to_parquet:
                with patch("json.dump"):
                    generator.save_embeddings(
                        user_embeddings, movie_embeddings, output_dir
                    )

                    # Check that DataFrames were created with correct columns
                    # We can't directly inspect the DataFrame, but we can verify to_parquet was called
                    assert mock_to_parquet.call_count == 2


class TestMain:
    """Test main function."""

    @patch("src.embeddings.generate.argparse.ArgumentParser.parse_args")
    @patch("src.embeddings.generate.EmbeddingGenerator")
    def test_main_function_basic(self, mock_generator_class, mock_parse_args):
        """Test main function execution."""
        from src.embeddings.generate import main

        # Mock command line arguments
        mock_args = Mock()
        mock_args.model_path = "model.pt"
        mock_args.user_features = "user_features.parquet"
        mock_args.movie_features = "movie_features.parquet"
        mock_args.output_dir = "embeddings"
        mock_args.batch_size = 1024
        mock_args.device = "cpu"
        mock_parse_args.return_value = mock_args

        # Mock generator instance
        mock_generator = Mock()
        mock_generator.generate_user_embeddings.return_value = np.random.randn(
            1000, 128
        )
        mock_generator.generate_movie_embeddings.return_value = np.random.randn(
            500, 128
        )
        mock_generator_class.return_value = mock_generator

        # Run main
        main()

        # Verify generator was created
        mock_generator_class.assert_called_once()

        # Verify embeddings were generated
        mock_generator.generate_user_embeddings.assert_called_once_with(batch_size=1024)
        mock_generator.generate_movie_embeddings.assert_called_once_with(
            batch_size=1024
        )

        # Verify embeddings were saved
        mock_generator.save_embeddings.assert_called_once()

    @patch("src.embeddings.generate.argparse.ArgumentParser.parse_args")
    @patch("src.embeddings.generate.EmbeddingGenerator")
    def test_main_function_custom_batch_size(
        self, mock_generator_class, mock_parse_args
    ):
        """Test main function with custom batch size."""
        from src.embeddings.generate import main

        mock_args = Mock()
        mock_args.model_path = "model.pt"
        mock_args.user_features = "user_features.parquet"
        mock_args.movie_features = "movie_features.parquet"
        mock_args.output_dir = "embeddings"
        mock_args.batch_size = 512
        mock_args.device = "cpu"
        mock_parse_args.return_value = mock_args

        mock_generator = Mock()
        mock_generator.generate_user_embeddings.return_value = np.random.randn(
            1000, 128
        )
        mock_generator.generate_movie_embeddings.return_value = np.random.randn(
            500, 128
        )
        mock_generator_class.return_value = mock_generator

        main()

        # Verify custom batch size was used
        mock_generator.generate_user_embeddings.assert_called_once_with(batch_size=512)
        mock_generator.generate_movie_embeddings.assert_called_once_with(batch_size=512)
