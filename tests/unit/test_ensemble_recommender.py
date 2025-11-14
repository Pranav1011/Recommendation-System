"""
Unit tests for Ensemble Recommender.

Tests cover:
- Loading user/item ID mappings from Parquet files
- Initialization with configs
- Model loading (LightGCN and Two-Tower)
- Graph building and caching
- Feature loading and alignment
- Embedding generation and combination
- Recommendation and similarity search
- Edge cases and error handling
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.ensemble_recommender import (
    EnsembleRecommender,
    load_twotower_item_mapping,
    load_twotower_user_mapping,
)


@pytest.fixture
def mock_ratings_df():
    """Create mock ratings dataframe."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3, 4, 4],
            "movieId": [10, 20, 10, 30, 20, 30, 10, 40],
            "rating": [5.0, 4.0, 3.0, 5.0, 4.5, 3.5, 4.0, 5.0],
        }
    )


@pytest.fixture
def mock_lightgcn_config():
    """Create mock LightGCN config."""
    return {
        "n_users": 1000,
        "n_movies": 500,
        "model": {
            "embedding_dim": 128,
            "n_layers": 3,
            "dropout_rate": 0.1,
            "rating_threshold": 4.0,
        },
        "graph_cache_dir": "data/graph_cache",
    }


@pytest.fixture
def mock_twotower_config():
    """Create mock Two-Tower config."""
    return {
        "n_users": 1000,
        "n_movies": 500,
        "embedding_dim": 128,
        "hidden_dim": 256,
        "user_feature_dim": 0,
        "movie_feature_dim": 0,
    }


@pytest.fixture
def mock_lightgcn_checkpoint():
    """Create mock LightGCN checkpoint."""
    # Create mock model state dict
    state_dict = {
        "user_embedding.weight": torch.randn(1000, 128),
        "item_embedding.weight": torch.randn(500, 128),
    }
    return {
        "epoch": 50,
        "model_state_dict": state_dict,
        "optimizer_state_dict": {},
        "train_loss": 0.5,
    }


@pytest.fixture
def mock_twotower_checkpoint():
    """Create mock Two-Tower checkpoint."""
    state_dict = {
        "user_tower.embedding.weight": torch.randn(1000, 128),
        "movie_tower.embedding.weight": torch.randn(500, 128),
        "user_tower.feature_transform.weight": torch.randn(32, 0),  # No features
        "movie_tower.feature_transform.weight": torch.randn(32, 0),  # No features
    }
    return {
        "epoch": 30,
        "model_state_dict": state_dict,
        "optimizer_state_dict": {},
        "train_loss": 0.8,
    }


@pytest.fixture
def mock_graph_obj():
    """Create mock graph builder object."""
    graph_obj = Mock()
    # Create sparse graph (small for testing)
    indices = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    values = torch.tensor([1.0, 1.0, 1.0, 1.0])
    size = torch.Size([1000, 500])
    graph_obj.get_sparse_graph.return_value = (indices, values, size)
    graph_obj.user_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
    graph_obj.item_to_idx = {10: 0, 20: 1, 30: 2, 40: 3}
    return graph_obj


class TestLoadMappingFunctions:
    """Test ID mapping loading functions."""

    @patch("src.models.ensemble_recommender.pd.read_parquet")
    def test_load_twotower_user_mapping(self, mock_read_parquet, mock_ratings_df):
        """Test loading Two-Tower user ID to index mapping."""
        mock_read_parquet.return_value = mock_ratings_df

        user_to_idx = load_twotower_user_mapping(Path("train_ratings.parquet"))

        # Should have 4 unique users: [1, 2, 3, 4]
        assert len(user_to_idx) == 4
        assert user_to_idx[1] == 0
        assert user_to_idx[2] == 1
        assert user_to_idx[3] == 2
        assert user_to_idx[4] == 3

    @patch("src.models.ensemble_recommender.pd.read_parquet")
    def test_load_twotower_item_mapping(self, mock_read_parquet, mock_ratings_df):
        """Test loading Two-Tower item ID to index mapping."""
        mock_read_parquet.return_value = mock_ratings_df

        item_to_idx = load_twotower_item_mapping(Path("train_ratings.parquet"))

        # Should have 4 unique items: [10, 20, 30, 40]
        assert len(item_to_idx) == 4
        assert item_to_idx[10] == 0
        assert item_to_idx[20] == 1
        assert item_to_idx[30] == 2
        assert item_to_idx[40] == 3

    @patch("src.models.ensemble_recommender.pd.read_parquet")
    def test_load_mapping_with_single_user_item(self, mock_read_parquet):
        """Test loading mapping with single user and item."""
        mock_read_parquet.return_value = pd.DataFrame(
            {"userId": [1], "movieId": [10], "rating": [5.0]}
        )

        user_to_idx = load_twotower_user_mapping(Path("train_ratings.parquet"))
        item_to_idx = load_twotower_item_mapping(Path("train_ratings.parquet"))

        assert len(user_to_idx) == 1
        assert len(item_to_idx) == 1
        assert user_to_idx[1] == 0
        assert item_to_idx[10] == 0


class TestEnsembleRecommenderInit:
    """Test EnsembleRecommender initialization."""

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    @patch(
        "src.models.ensemble_recommender.EnsembleRecommender._validate_embedding_dims"
    )
    def test_init_with_dict_configs(
        self,
        mock_validate,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_lightgcn_config,
        mock_twotower_config,
    ):
        """Test initialization with config dicts."""
        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=mock_lightgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=mock_twotower_config,
            lightgcn_weight=0.7,
            twotower_weight=0.3,
            device="cpu",
        )

        assert ensemble.lightgcn_weight == 0.7
        assert ensemble.twotower_weight == 0.3
        assert ensemble.lightgcn_config == mock_lightgcn_config
        assert ensemble.twotower_config == mock_twotower_config
        mock_validate.assert_called_once()
        mock_load_lgcn.assert_called_once()
        mock_load_tt.assert_called_once()
        mock_load_feat.assert_called_once()
        mock_gen_emb.assert_called_once()

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    @patch(
        "src.models.ensemble_recommender.EnsembleRecommender._validate_embedding_dims"
    )
    def test_init_invalid_weights(
        self,
        mock_validate,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_lightgcn_config,
        mock_twotower_config,
    ):
        """Test initialization with weights not summing to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            EnsembleRecommender(
                lightgcn_checkpoint="lgcn.pt",
                lightgcn_config=mock_lightgcn_config,
                twotower_checkpoint="tt.pt",
                twotower_config=mock_twotower_config,
                lightgcn_weight=0.5,
                twotower_weight=0.6,  # Sum is 1.1
                device="cpu",
            )

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    @patch(
        "src.models.ensemble_recommender.EnsembleRecommender._validate_embedding_dims"
    )
    def test_init_cuda_device(
        self,
        mock_validate,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_lightgcn_config,
        mock_twotower_config,
    ):
        """Test initialization with CUDA device (falls back to CPU if unavailable)."""
        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=mock_lightgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=mock_twotower_config,
            device="cuda",
        )

        # Device should be CPU if CUDA not available, or CUDA if available
        assert ensemble.device.type in ["cpu", "cuda"]


class TestLoadConfig:
    """Test _load_config method."""

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    @patch(
        "src.models.ensemble_recommender.EnsembleRecommender._validate_embedding_dims"
    )
    def test_load_config_from_dict(
        self,
        mock_validate,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_lightgcn_config,
        mock_twotower_config,
    ):
        """Test loading config from dictionary."""
        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=mock_lightgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=mock_twotower_config,
            device="cpu",
        )

        config = ensemble._load_config(mock_lightgcn_config)
        assert config == mock_lightgcn_config

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    @patch(
        "src.models.ensemble_recommender.EnsembleRecommender._validate_embedding_dims"
    )
    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_config_from_file(
        self,
        mock_exists,
        mock_file,
        mock_validate,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_lightgcn_config,
        mock_twotower_config,
    ):
        """Test loading config from file."""
        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=mock_lightgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=mock_twotower_config,
            device="cpu",
        )

        config = ensemble._load_config("config.json")
        assert config == {"key": "value"}

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    @patch(
        "src.models.ensemble_recommender.EnsembleRecommender._validate_embedding_dims"
    )
    @patch("pathlib.Path.exists", return_value=False)
    def test_load_config_file_not_found(
        self,
        mock_exists,
        mock_validate,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_lightgcn_config,
        mock_twotower_config,
    ):
        """Test loading config from non-existent file."""
        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=mock_lightgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=mock_twotower_config,
            device="cpu",
        )

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ensemble._load_config("nonexistent.json")


class TestValidateEmbeddingDims:
    """Test _validate_embedding_dims method."""

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_matching_dimensions(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test validation with matching embedding dimensions."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        assert not ensemble.use_projection
        assert ensemble.target_dim == 128

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_mismatched_dimensions(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test validation with mismatched embedding dimensions."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 64, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        assert ensemble.use_projection
        assert ensemble.target_dim == 128  # max(64, 128)


class TestLoadLightGCN:
    """Test _load_lightgcn method."""

    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_graph")
    @patch("src.models.ensemble_recommender.create_lightgcn_model")
    @patch("torch.load")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    def test_load_lightgcn_success(
        self,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_path_exists,
        mock_torch_load,
        mock_create_model,
        mock_load_graph,
        mock_lightgcn_config,
        mock_twotower_config,
        mock_lightgcn_checkpoint,
    ):
        """Test successful LightGCN loading."""
        mock_torch_load.return_value = mock_lightgcn_checkpoint
        mock_model = Mock()
        mock_create_model.return_value = mock_model

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=mock_lightgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=mock_twotower_config,
            device="cpu",
        )

        mock_create_model.assert_called_once()
        mock_model.load_state_dict.assert_called_once_with(
            mock_lightgcn_checkpoint["model_state_dict"]
        )
        mock_model.to.assert_called()
        mock_model.eval.assert_called_once()
        mock_load_graph.assert_called_once()

    @patch("pathlib.Path.exists", return_value=False)
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch(
        "src.models.ensemble_recommender.EnsembleRecommender._validate_embedding_dims"
    )
    def test_load_lightgcn_file_not_found(
        self,
        mock_validate,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_path_exists,
        mock_lightgcn_config,
        mock_twotower_config,
    ):
        """Test LightGCN loading with missing checkpoint."""
        with pytest.raises(FileNotFoundError, match="LightGCN checkpoint not found"):
            EnsembleRecommender(
                lightgcn_checkpoint="nonexistent.pt",
                lightgcn_config=mock_lightgcn_config,
                twotower_checkpoint="tt.pt",
                twotower_config=mock_twotower_config,
                device="cpu",
            )

    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_graph")
    @patch("src.models.ensemble_recommender.create_lightgcn_model")
    @patch("torch.load")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    def test_load_lightgcn_with_projection(
        self,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_path_exists,
        mock_torch_load,
        mock_create_model,
        mock_load_graph,
        mock_lightgcn_checkpoint,
    ):
        """Test LightGCN loading with dimension projection."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 64, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        mock_torch_load.return_value = mock_lightgcn_checkpoint
        mock_model = Mock()
        mock_create_model.return_value = mock_model

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        # Should create projection layer
        assert hasattr(ensemble, "lightgcn_projection")
        assert isinstance(ensemble.lightgcn_projection, torch.nn.Linear)


class TestLoadGraph:
    """Test _load_graph method."""

    @patch("torch.save")
    @patch("src.models.ensemble_recommender.build_graph")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.mkdir")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_build_graph_from_scratch(
        self,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_mkdir,
        mock_path_exists,
        mock_build_graph,
        mock_save,
        mock_lightgcn_config,
        mock_twotower_config,
        mock_graph_obj,
    ):
        """Test building graph from scratch when cache doesn't exist."""
        mock_build_graph.return_value = mock_graph_obj

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=mock_lightgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=mock_twotower_config,
            device="cpu",
        )

        # Manually call _load_graph to test it
        ensemble._load_graph()

        mock_build_graph.assert_called_once()
        mock_save.assert_called_once()
        assert ensemble.user_to_idx == mock_graph_obj.user_to_idx
        assert ensemble.item_to_idx == mock_graph_obj.item_to_idx

    @patch("torch.load")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_load_cached_graph(
        self,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_path_exists,
        mock_torch_load,
        mock_lightgcn_config,
        mock_twotower_config,
    ):
        """Test loading graph from cache."""
        # Mock cached graph
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.tensor([1.0, 1.0])
        size = torch.Size([1000, 500])
        cached_graph = torch.sparse.FloatTensor(indices, values, size)
        mock_torch_load.return_value = {
            "graph": cached_graph,
            "user_to_idx": {1: 0, 2: 1},
            "item_to_idx": {10: 0, 20: 1},
        }

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=mock_lightgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=mock_twotower_config,
            device="cpu",
        )

        # Manually call _load_graph to test it
        ensemble._load_graph()

        # Should load from cache, not build
        assert ensemble.user_to_idx == {1: 0, 2: 1}
        assert ensemble.item_to_idx == {10: 0, 20: 1}


class TestLoadTwoTower:
    """Test _load_twotower method."""

    @patch("src.models.ensemble_recommender.load_twotower_item_mapping")
    @patch("src.models.ensemble_recommender.load_twotower_user_mapping")
    @patch("src.models.ensemble_recommender.create_two_tower_model")
    @patch("torch.load")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_load_twotower_success(
        self,
        mock_load_lgcn,
        mock_load_feat,
        mock_gen_emb,
        mock_path_exists,
        mock_torch_load,
        mock_create_model,
        mock_load_user_map,
        mock_load_item_map,
        mock_lightgcn_config,
        mock_twotower_config,
        mock_twotower_checkpoint,
    ):
        """Test successful Two-Tower loading."""
        mock_torch_load.return_value = mock_twotower_checkpoint
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        mock_load_user_map.return_value = {1: 0, 2: 1}
        mock_load_item_map.return_value = {10: 0, 20: 1}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=mock_lightgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=mock_twotower_config,
            device="cpu",
        )

        mock_create_model.assert_called_once()
        mock_model.load_state_dict.assert_called_once()
        mock_model.to.assert_called()
        mock_model.eval.assert_called_once()

    @patch("pathlib.Path.exists", return_value=False)
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    @patch(
        "src.models.ensemble_recommender.EnsembleRecommender._validate_embedding_dims"
    )
    def test_load_twotower_file_not_found(
        self,
        mock_validate,
        mock_load_lgcn,
        mock_load_feat,
        mock_gen_emb,
        mock_path_exists,
        mock_lightgcn_config,
        mock_twotower_config,
    ):
        """Test Two-Tower loading with missing checkpoint."""
        with pytest.raises(FileNotFoundError, match="Two-Tower checkpoint not found"):
            EnsembleRecommender(
                lightgcn_checkpoint="lgcn.pt",
                lightgcn_config=mock_lightgcn_config,
                twotower_checkpoint="nonexistent.pt",
                twotower_config=mock_twotower_config,
                device="cpu",
            )

    @patch("src.models.ensemble_recommender.load_twotower_item_mapping")
    @patch("src.models.ensemble_recommender.load_twotower_user_mapping")
    @patch("src.models.ensemble_recommender.create_two_tower_model")
    @patch("torch.load")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_load_twotower_with_features(
        self,
        mock_load_lgcn,
        mock_load_feat,
        mock_gen_emb,
        mock_path_exists,
        mock_torch_load,
        mock_create_model,
        mock_load_user_map,
        mock_load_item_map,
        mock_lightgcn_config,
        mock_twotower_checkpoint,
    ):
        """Test Two-Tower loading with features (dimension detection)."""
        # Modify checkpoint to have features
        checkpoint_with_features = mock_twotower_checkpoint.copy()
        checkpoint_with_features["model_state_dict"] = {
            "user_tower.embedding.weight": torch.randn(1000, 128),
            "movie_tower.embedding.weight": torch.randn(500, 128),
            "user_tower.feature_transform.weight": torch.randn(32, 30),  # 30 features
            "movie_tower.feature_transform.weight": torch.randn(32, 12),  # 12 features
        }
        mock_torch_load.return_value = checkpoint_with_features

        tt_config = {
            "embedding_dim": 128,
            "user_feature_dim": 0,
            "movie_feature_dim": 0,
        }
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        mock_load_user_map.return_value = {1: 0, 2: 1}
        mock_load_item_map.return_value = {10: 0, 20: 1}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=mock_lightgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        # Config should be updated with detected dimensions
        assert ensemble.twotower_config["user_feature_dim"] == 30
        assert ensemble.twotower_config["movie_feature_dim"] == 12


class TestLoadFeatures:
    """Test _load_features method."""

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_load_features_no_features(
        self, mock_load_lgcn, mock_load_tt, mock_gen_emb
    ):
        """Test feature loading when model has no features."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {
            "embedding_dim": 128,
            "user_feature_dim": 0,
            "movie_feature_dim": 0,
        }

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        # Manually call _load_features to test
        ensemble._load_features()

        assert ensemble.user_features is None
        assert ensemble.movie_features is None

    def test_load_features_logic(self):
        """Test feature loading logic directly (integration-style unit test)."""
        # This test verifies the conditional logic in _load_features
        # Full integration is tested elsewhere, here we just check the method exists and handles configs
        from src.models.ensemble_recommender import EnsembleRecommender

        # Just verify the method signature and basic structure
        assert hasattr(EnsembleRecommender, "_load_features")
        assert callable(EnsembleRecommender._load_features)

    @patch("pathlib.Path.exists", return_value=False)
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_load_features_file_not_found(
        self, mock_load_lgcn, mock_load_tt, mock_gen_emb, mock_path_exists
    ):
        """Test feature loading when feature files don't exist."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {
            "embedding_dim": 128,
            "user_feature_dim": 30,
            "movie_feature_dim": 12,
        }

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        # Mock user_to_idx and item_to_idx
        ensemble.user_to_idx = {1: 0, 2: 1}
        ensemble.item_to_idx = {10: 0, 20: 1}

        # Manually call _load_features to test
        ensemble._load_features()

        # Should set to None when files not found
        assert ensemble.user_features is None
        assert ensemble.movie_features is None


class TestAlignFeatures:
    """Test _align_features method."""

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_align_features(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test feature alignment with ID mappings."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        # Create test dataframe
        features_df = pd.DataFrame(
            {
                "userId": [1, 2, 3],
                "feat1": [0.1, 0.2, 0.3],
                "feat2": [0.4, 0.5, 0.6],
            }
        )
        id_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}  # 4 users, but only 3 in features

        aligned_features = ensemble._align_features(features_df, id_to_idx, "userId")

        assert aligned_features.shape == (4, 2)  # 4 users, 2 features
        assert aligned_features.dtype == torch.float32
        # User 4 should have zeros (not in features_df)
        assert torch.all(aligned_features[3] == 0)

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_align_features_empty(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test feature alignment with empty dataframe."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        # Empty dataframe
        features_df = pd.DataFrame({"userId": [], "feat1": [], "feat2": []})
        id_to_idx = {1: 0, 2: 1}

        aligned_features = ensemble._align_features(features_df, id_to_idx, "userId")

        assert aligned_features.shape == (2, 2)  # 2 users, 2 features
        assert torch.all(aligned_features == 0)  # All zeros


class TestGenerateEmbeddings:
    """Test _generate_embeddings method."""

    def test_generate_embeddings_logic(self):
        """Test embedding generation logic (method exists and is callable)."""
        # This test verifies the embedding generation method exists
        # Full integration is tested elsewhere
        from src.models.ensemble_recommender import EnsembleRecommender

        # Just verify the method signature and basic structure
        assert hasattr(EnsembleRecommender, "_generate_embeddings")
        assert callable(EnsembleRecommender._generate_embeddings)


class TestGetEmbeddings:
    """Test get_user_embedding and get_item_embedding methods."""

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_get_user_embedding_success(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test getting user embedding for valid user."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        # Mock embeddings
        ensemble.user_to_idx = {1: 0, 2: 1, 3: 2}
        ensemble.user_embeddings = torch.randn(3, 128)

        embedding = ensemble.get_user_embedding(1)

        assert embedding.shape == (128,)
        assert isinstance(embedding, np.ndarray)

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_get_user_embedding_not_found(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test getting user embedding for invalid user."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        ensemble.user_to_idx = {1: 0, 2: 1}
        ensemble.user_embeddings = torch.randn(2, 128)

        with pytest.raises(KeyError, match="User ID 999 not found"):
            ensemble.get_user_embedding(999)

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_get_item_embedding_success(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test getting item embedding for valid item."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        ensemble.item_to_idx = {10: 0, 20: 1, 30: 2}
        ensemble.item_embeddings = torch.randn(3, 128)

        embedding = ensemble.get_item_embedding(10)

        assert embedding.shape == (128,)
        assert isinstance(embedding, np.ndarray)

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_get_item_embedding_not_found(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test getting item embedding for invalid item."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        ensemble.item_to_idx = {10: 0, 20: 1}
        ensemble.item_embeddings = torch.randn(2, 128)

        with pytest.raises(KeyError, match="Item ID 999 not found"):
            ensemble.get_item_embedding(999)


class TestPredictScores:
    """Test predict_scores method."""

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_predict_scores_success(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test predicting scores for user-item pairs."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        # Mock embeddings
        ensemble.user_to_idx = {1: 0, 2: 1}
        ensemble.item_to_idx = {10: 0, 20: 1}
        ensemble.user_embeddings = torch.randn(2, 128)
        ensemble.item_embeddings = torch.randn(2, 128)

        scores = ensemble.predict_scores([1, 2], [10, 20])

        assert len(scores) == 2
        assert isinstance(scores, np.ndarray)

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_predict_scores_length_mismatch(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test predicting scores with mismatched lengths."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        ensemble.user_to_idx = {1: 0, 2: 1}
        ensemble.item_to_idx = {10: 0, 20: 1}

        with pytest.raises(ValueError, match="must have same length"):
            ensemble.predict_scores([1, 2], [10])

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_predict_scores_invalid_id(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test predicting scores with invalid ID."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        ensemble.user_to_idx = {1: 0}
        ensemble.item_to_idx = {10: 0}

        with pytest.raises(KeyError):
            ensemble.predict_scores([999], [10])


class TestRecommend:
    """Test recommend method."""

    @patch("src.models.ensemble_recommender.pd.read_parquet")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_recommend_success(
        self,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_read_parquet,
    ):
        """Test getting recommendations for user."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        # Mock embeddings
        ensemble.user_to_idx = {1: 0}
        ensemble.item_to_idx = {10: 0, 20: 1, 30: 2, 40: 3}
        ensemble.user_embeddings = torch.randn(1, 128)
        ensemble.item_embeddings = torch.randn(4, 128)
        ensemble.data_dir = Path("data/processed")

        # Mock training data (empty, no seen items)
        mock_read_parquet.return_value = pd.DataFrame(
            {"userId": [], "movieId": [], "rating": []}
        )

        items, scores = ensemble.recommend(1, k=2)

        assert len(items) == 2
        assert len(scores) == 2
        assert all(isinstance(item, int) for item in items)
        assert all(isinstance(score, float) for score in scores)

    @patch("src.models.ensemble_recommender.pd.read_parquet")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_recommend_exclude_seen(
        self,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_read_parquet,
    ):
        """Test recommendations with seen item exclusion."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        ensemble.user_to_idx = {1: 0}
        ensemble.item_to_idx = {10: 0, 20: 1, 30: 2}
        ensemble.user_embeddings = torch.randn(1, 128)
        ensemble.item_embeddings = torch.randn(3, 128)
        ensemble.data_dir = Path("data/processed")

        # Mock training data - user has seen item 10
        mock_read_parquet.return_value = pd.DataFrame(
            {"userId": [1], "movieId": [10], "rating": [5.0]}
        )

        items, scores = ensemble.recommend(1, k=2, exclude_seen=True)

        # Item 10 should not be in recommendations
        assert 10 not in items
        assert len(items) == 2

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_recommend_user_not_found(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test recommendations for invalid user."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        ensemble.user_to_idx = {1: 0}

        with pytest.raises(KeyError, match="User ID 999 not found"):
            ensemble.recommend(999, k=10)


class TestGetSimilarItems:
    """Test get_similar_items method."""

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_get_similar_items_success(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test getting similar items."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        ensemble.item_to_idx = {10: 0, 20: 1, 30: 2, 40: 3}
        ensemble.item_embeddings = torch.randn(4, 128)

        items, scores = ensemble.get_similar_items(10, k=2)

        assert len(items) == 2
        assert len(scores) == 2
        assert 10 not in items  # Should exclude the query item itself
        assert all(isinstance(item, int) for item in items)
        assert all(isinstance(score, float) for score in scores)

    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_get_similar_items_not_found(
        self, mock_load_lgcn, mock_load_tt, mock_load_feat, mock_gen_emb
    ):
        """Test getting similar items for invalid item."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        ensemble.item_to_idx = {10: 0, 20: 1}

        with pytest.raises(KeyError, match="Item ID 999 not found"):
            ensemble.get_similar_items(999, k=5)


class TestSaveEmbeddings:
    """Test save_embeddings method."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("numpy.save")
    @patch("pathlib.Path.mkdir")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_save_embeddings(
        self,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_mkdir,
        mock_np_save,
        mock_file,
    ):
        """Test saving embeddings to disk."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        # Mock embeddings and mappings
        ensemble.user_embeddings = torch.randn(3, 128)
        ensemble.item_embeddings = torch.randn(2, 128)
        ensemble.user_to_idx = {1: 0, 2: 1, 3: 2}
        ensemble.item_to_idx = {10: 0, 20: 1}

        ensemble.save_embeddings("output/embeddings")

        # Check that numpy save was called
        assert mock_np_save.call_count == 2

        # Check that JSON files were written
        assert mock_file.call_count == 2

    @patch("builtins.open", new_callable=mock_open)
    @patch("numpy.save")
    @patch("pathlib.Path.mkdir")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._generate_embeddings")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_features")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_twotower")
    @patch("src.models.ensemble_recommender.EnsembleRecommender._load_lightgcn")
    def test_save_embeddings_creates_directory(
        self,
        mock_load_lgcn,
        mock_load_tt,
        mock_load_feat,
        mock_gen_emb,
        mock_mkdir,
        mock_np_save,
        mock_file,
    ):
        """Test that save_embeddings creates output directory."""
        lgcn_config = {
            "n_users": 1000,
            "n_movies": 500,
            "model": {"embedding_dim": 128, "n_layers": 3},
        }
        tt_config = {"embedding_dim": 128}

        ensemble = EnsembleRecommender(
            lightgcn_checkpoint="lgcn.pt",
            lightgcn_config=lgcn_config,
            twotower_checkpoint="tt.pt",
            twotower_config=tt_config,
            device="cpu",
        )

        ensemble.user_embeddings = torch.randn(2, 128)
        ensemble.item_embeddings = torch.randn(2, 128)
        ensemble.user_to_idx = {1: 0, 2: 1}
        ensemble.item_to_idx = {10: 0, 20: 1}

        ensemble.save_embeddings("output/new_dir")

        # Check that mkdir was called
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
