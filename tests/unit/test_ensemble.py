"""
Unit tests for Ensemble Recommender System.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.ensemble import EnsembleRecommender, create_ensemble


class TestEnsembleRecommenderInit:
    """Test EnsembleRecommender initialization."""

    @patch.object(EnsembleRecommender, "_load_all_embeddings")
    def test_init_default_weights(self, mock_load):
        """Test initialization with default equal weights."""
        configs = {
            "model1": {"embeddings_dir": "dir1"},
            "model2": {"embeddings_dir": "dir2"},
        }

        recommender = EnsembleRecommender(configs)

        assert recommender.weights == {"model1": 0.5, "model2": 0.5}
        assert recommender.combination_method == "weighted_score"
        mock_load.assert_called_once()

    @patch.object(EnsembleRecommender, "_load_all_embeddings")
    def test_init_custom_weights(self, mock_load):
        """Test initialization with custom weights."""
        configs = {
            "model1": {"embeddings_dir": "dir1"},
            "model2": {"embeddings_dir": "dir2"},
        }
        weights = {"model1": 0.7, "model2": 0.3}

        recommender = EnsembleRecommender(configs, ensemble_weights=weights)

        assert recommender.weights["model1"] == pytest.approx(0.7)
        assert recommender.weights["model2"] == pytest.approx(0.3)

    @patch.object(EnsembleRecommender, "_load_all_embeddings")
    def test_init_weights_normalization(self, mock_load):
        """Test that weights are normalized to sum to 1."""
        configs = {
            "model1": {"embeddings_dir": "dir1"},
            "model2": {"embeddings_dir": "dir2"},
        }
        # Non-normalized weights
        weights = {"model1": 2.0, "model2": 3.0}

        recommender = EnsembleRecommender(configs, ensemble_weights=weights)

        assert recommender.weights["model1"] == pytest.approx(0.4)
        assert recommender.weights["model2"] == pytest.approx(0.6)
        assert sum(recommender.weights.values()) == pytest.approx(1.0)

    @patch.object(EnsembleRecommender, "_load_all_embeddings")
    def test_init_combination_methods(self, mock_load):
        """Test different combination methods."""
        configs = {"model1": {"embeddings_dir": "dir1"}}

        for method in ["weighted_score", "weighted_rank", "borda_count"]:
            recommender = EnsembleRecommender(configs, combination_method=method)
            assert recommender.combination_method == method


class TestEnsembleEmbeddingLoading:
    """Test embedding loading functionality."""

    def test_load_embeddings_success(self):
        """Test successful embedding loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock embedding files
            user_emb = np.random.rand(100, 64).astype(np.float32)
            movie_emb = np.random.rand(500, 64).astype(np.float32)

            np.save(Path(tmpdir) / "user_embeddings.npy", user_emb)
            np.save(Path(tmpdir) / "movie_embeddings.npy", movie_emb)

            configs = {"model1": {"embeddings_dir": tmpdir}}

            recommender = EnsembleRecommender(configs)

            assert "model1" in recommender.embeddings
            assert recommender.embeddings["model1"]["user"].shape == (100, 64)
            assert recommender.embeddings["model1"]["movie"].shape == (500, 64)

    def test_load_embeddings_file_not_found(self):
        """Test error when embedding files don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = {"model1": {"embeddings_dir": tmpdir}}

            with pytest.raises(FileNotFoundError):
                EnsembleRecommender(configs)


class TestEnsembleRecommendations:
    """Test recommendation generation."""

    @pytest.fixture
    def ensemble_with_embeddings(self):
        """Create ensemble with mock embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create normalized embeddings for consistent similarity scores
            np.random.seed(42)
            user_emb = np.random.rand(100, 64).astype(np.float32)
            user_emb = user_emb / np.linalg.norm(user_emb, axis=1, keepdims=True)
            movie_emb = np.random.rand(50, 64).astype(np.float32)
            movie_emb = movie_emb / np.linalg.norm(movie_emb, axis=1, keepdims=True)

            np.save(Path(tmpdir) / "user_embeddings.npy", user_emb)
            np.save(Path(tmpdir) / "movie_embeddings.npy", movie_emb)

            configs = {"model1": {"embeddings_dir": tmpdir}}
            yield EnsembleRecommender(configs)

    def test_weighted_score_recommendations(self, ensemble_with_embeddings):
        """Test weighted score combination."""
        user_indices = [0, 1, 2]
        recommendations, scores = ensemble_with_embeddings.get_recommendations(
            user_indices, k=10, return_scores=True
        )

        assert len(recommendations) == 3
        assert all(len(recs) == 10 for recs in recommendations)
        assert scores is not None
        assert len(scores) == 3
        assert all(len(s) == 10 for s in scores)

    def test_weighted_rank_recommendations(self):
        """Test weighted rank combination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.random.seed(42)
            user_emb = np.random.rand(50, 32).astype(np.float32)
            movie_emb = np.random.rand(30, 32).astype(np.float32)

            np.save(Path(tmpdir) / "user_embeddings.npy", user_emb)
            np.save(Path(tmpdir) / "movie_embeddings.npy", movie_emb)

            configs = {"model1": {"embeddings_dir": tmpdir}}
            recommender = EnsembleRecommender(
                configs, combination_method="weighted_rank"
            )

            recommendations, scores = recommender.get_recommendations(
                [0, 1], k=5, return_scores=True
            )

            assert len(recommendations) == 2
            assert all(len(recs) == 5 for recs in recommendations)
            assert scores is not None

    def test_borda_count_recommendations(self):
        """Test Borda count combination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.random.seed(42)
            user_emb = np.random.rand(50, 32).astype(np.float32)
            movie_emb = np.random.rand(30, 32).astype(np.float32)

            np.save(Path(tmpdir) / "user_embeddings.npy", user_emb)
            np.save(Path(tmpdir) / "movie_embeddings.npy", movie_emb)

            configs = {"model1": {"embeddings_dir": tmpdir}}
            recommender = EnsembleRecommender(configs, combination_method="borda_count")

            recommendations, scores = recommender.get_recommendations(
                [0, 1], k=5, return_scores=True
            )

            assert len(recommendations) == 2
            assert all(len(recs) == 5 for recs in recommendations)
            assert scores is not None

    def test_invalid_combination_method(self):
        """Test error for invalid combination method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.random.seed(42)
            user_emb = np.random.rand(10, 16).astype(np.float32)
            movie_emb = np.random.rand(20, 16).astype(np.float32)

            np.save(Path(tmpdir) / "user_embeddings.npy", user_emb)
            np.save(Path(tmpdir) / "movie_embeddings.npy", movie_emb)

            configs = {"model1": {"embeddings_dir": tmpdir}}
            recommender = EnsembleRecommender(
                configs, combination_method="invalid_method"
            )

            with pytest.raises(ValueError, match="Unknown combination method"):
                recommender.get_recommendations([0], k=5)

    def test_recommendations_without_scores(self, ensemble_with_embeddings):
        """Test getting recommendations without scores."""
        recommendations, scores = ensemble_with_embeddings.get_recommendations(
            [0, 1], k=5, return_scores=False
        )

        assert len(recommendations) == 2
        assert scores is None


class TestEnsembleMultipleModels:
    """Test ensemble with multiple models."""

    def test_two_model_ensemble(self):
        """Test ensemble combining two models."""
        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            np.random.seed(42)
            # Model 1 embeddings
            user_emb1 = np.random.rand(50, 32).astype(np.float32)
            movie_emb1 = np.random.rand(30, 32).astype(np.float32)
            np.save(Path(tmpdir1) / "user_embeddings.npy", user_emb1)
            np.save(Path(tmpdir1) / "movie_embeddings.npy", movie_emb1)

            # Model 2 embeddings
            user_emb2 = np.random.rand(50, 32).astype(np.float32)
            movie_emb2 = np.random.rand(30, 32).astype(np.float32)
            np.save(Path(tmpdir2) / "user_embeddings.npy", user_emb2)
            np.save(Path(tmpdir2) / "movie_embeddings.npy", movie_emb2)

            configs = {
                "model1": {"embeddings_dir": tmpdir1},
                "model2": {"embeddings_dir": tmpdir2},
            }
            weights = {"model1": 0.6, "model2": 0.4}

            recommender = EnsembleRecommender(configs, ensemble_weights=weights)

            recommendations, scores = recommender.get_recommendations(
                [0, 1, 2], k=10, return_scores=True
            )

            assert len(recommendations) == 3
            assert len(recommender.embeddings) == 2


class TestAnalyzeModelContributions:
    """Test model contribution analysis."""

    def test_analyze_contributions(self):
        """Test model contribution analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.random.seed(42)
            user_emb = np.random.rand(20, 16).astype(np.float32)
            movie_emb = np.random.rand(30, 16).astype(np.float32)

            np.save(Path(tmpdir) / "user_embeddings.npy", user_emb)
            np.save(Path(tmpdir) / "movie_embeddings.npy", movie_emb)

            configs = {"model1": {"embeddings_dir": tmpdir}}
            recommender = EnsembleRecommender(configs)

            contributions = recommender.analyze_model_contributions([0, 1, 2], k=5)

            assert "model1" in contributions
            assert "mean_overlap" in contributions["model1"]
            assert "std_overlap" in contributions["model1"]
            assert "weight" in contributions["model1"]
            # Single model should have 100% overlap with itself
            assert contributions["model1"]["mean_overlap"] == 1.0


class TestCreateEnsemble:
    """Test create_ensemble helper function."""

    def test_create_ensemble_basic(self):
        """Test create_ensemble helper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.random.seed(42)
            user_emb = np.random.rand(10, 16).astype(np.float32)
            movie_emb = np.random.rand(20, 16).astype(np.float32)

            np.save(Path(tmpdir) / "user_embeddings.npy", user_emb)
            np.save(Path(tmpdir) / "movie_embeddings.npy", movie_emb)

            model_dirs = {"model1": tmpdir}
            ensemble = create_ensemble(model_dirs)

            assert isinstance(ensemble, EnsembleRecommender)
            assert ensemble.combination_method == "weighted_score"

    def test_create_ensemble_with_options(self):
        """Test create_ensemble with custom options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.random.seed(42)
            user_emb = np.random.rand(10, 16).astype(np.float32)
            movie_emb = np.random.rand(20, 16).astype(np.float32)

            np.save(Path(tmpdir) / "user_embeddings.npy", user_emb)
            np.save(Path(tmpdir) / "movie_embeddings.npy", movie_emb)

            model_dirs = {"model1": tmpdir}
            weights = {"model1": 1.0}

            ensemble = create_ensemble(
                model_dirs, weights=weights, method="borda_count"
            )

            assert ensemble.combination_method == "borda_count"
            assert ensemble.weights["model1"] == 1.0
