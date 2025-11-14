"""
Unit tests for Two-Tower model.
"""

import pytest
import torch

from src.models.two_tower import MovieTower, TwoTowerModel, UserTower, create_model


class TestUserTower:
    """Test UserTower component."""

    def test_initialization(self):
        """Test UserTower initialization."""
        tower = UserTower(n_users=1000, embedding_dim=128)
        assert tower.n_users == 1000
        assert tower.embedding_dim == 128

    def test_forward_without_features(self):
        """Test forward pass without user features."""
        tower = UserTower(n_users=1000, embedding_dim=128)
        user_ids = torch.randint(0, 1000, (32,))

        embeddings = tower(user_ids)

        assert embeddings.shape == (32, 128)
        assert embeddings.dtype == torch.float32

    def test_forward_with_features(self):
        """Test forward pass with user features."""
        tower = UserTower(n_users=1000, embedding_dim=128, feature_dim=30)
        user_ids = torch.randint(0, 1000, (32,))
        user_features = torch.randn(32, 30)

        embeddings = tower(user_ids, user_features)

        assert embeddings.shape == (32, 128)
        assert embeddings.dtype == torch.float32

    def test_embedding_normalization(self):
        """Test that embeddings are L2 normalized."""
        tower = UserTower(n_users=1000, embedding_dim=128)
        user_ids = torch.randint(0, 1000, (32,))

        embeddings = tower(user_ids)

        # Check L2 norm is 1
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(32), atol=1e-5)


class TestMovieTower:
    """Test MovieTower component."""

    def test_initialization(self):
        """Test MovieTower initialization."""
        tower = MovieTower(n_movies=5000, embedding_dim=128)
        assert tower.n_movies == 5000
        assert tower.embedding_dim == 128

    def test_forward_without_features(self):
        """Test forward pass without movie features."""
        tower = MovieTower(n_movies=5000, embedding_dim=128)
        movie_ids = torch.randint(0, 5000, (32,))

        embeddings = tower(movie_ids)

        assert embeddings.shape == (32, 128)
        assert embeddings.dtype == torch.float32

    def test_forward_with_features(self):
        """Test forward pass with movie features."""
        tower = MovieTower(n_movies=5000, embedding_dim=128, feature_dim=12)
        movie_ids = torch.randint(0, 5000, (32,))
        movie_features = torch.randn(32, 12)

        embeddings = tower(movie_ids, movie_features)

        assert embeddings.shape == (32, 128)
        assert embeddings.dtype == torch.float32

    def test_embedding_normalization(self):
        """Test that embeddings are L2 normalized."""
        tower = MovieTower(n_movies=5000, embedding_dim=128)
        movie_ids = torch.randint(0, 5000, (32,))

        embeddings = tower(movie_ids)

        # Check L2 norm is 1
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(32), atol=1e-5)


class TestTwoTowerModel:
    """Test complete Two-Tower model."""

    def test_initialization(self):
        """Test model initialization."""
        model = TwoTowerModel(n_users=1000, n_movies=5000, embedding_dim=128)
        assert model.n_users == 1000
        assert model.n_movies == 5000
        assert model.embedding_dim == 128

    def test_forward_cosine_similarity(self):
        """Test forward pass with cosine similarity."""
        model = TwoTowerModel(
            n_users=1000, n_movies=5000, embedding_dim=128, similarity_method="cosine"
        )

        user_ids = torch.randint(0, 1000, (32,))
        movie_ids = torch.randint(0, 5000, (32,))

        ratings, user_emb, movie_emb = model(user_ids, movie_ids)

        assert ratings.shape == (32,)
        assert user_emb.shape == (32, 128)
        assert movie_emb.shape == (32, 128)

        # Check rating range [0.5, 5.0]
        assert (ratings >= 0.5).all()
        assert (ratings <= 5.0).all()

    def test_forward_dot_product(self):
        """Test forward pass with dot product similarity."""
        model = TwoTowerModel(
            n_users=1000,
            n_movies=5000,
            embedding_dim=128,
            similarity_method="dot_product",
        )

        user_ids = torch.randint(0, 1000, (32,))
        movie_ids = torch.randint(0, 5000, (32,))

        ratings, user_emb, movie_emb = model(user_ids, movie_ids)

        assert ratings.shape == (32,)
        assert user_emb.shape == (32, 128)
        assert movie_emb.shape == (32, 128)

    def test_get_user_embedding(self):
        """Test getting user embeddings."""
        model = TwoTowerModel(n_users=1000, n_movies=5000, embedding_dim=128)
        model.eval()  # Batch norm doesn't work with batch_size=1 in train mode

        user_id = torch.tensor([42])
        embedding = model.get_user_embedding(user_id)

        assert embedding.shape == (1, 128)

    def test_get_movie_embedding(self):
        """Test getting movie embeddings."""
        model = TwoTowerModel(n_users=1000, n_movies=5000, embedding_dim=128)
        model.eval()  # Batch norm doesn't work with batch_size=1 in train mode

        movie_id = torch.tensor([123])
        embedding = model.get_movie_embedding(movie_id)

        assert embedding.shape == (1, 128)

    def test_create_model_from_config(self):
        """Test model creation from config."""
        config = {
            "n_users": 1000,
            "n_movies": 5000,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "dropout_rate": 0.3,
        }

        model = create_model(config)

        assert model.n_users == 1000
        assert model.n_movies == 5000
        assert model.embedding_dim == 64

    def test_invalid_similarity_method(self):
        """Test that invalid similarity method raises error."""
        model = TwoTowerModel(n_users=1000, n_movies=5000, similarity_method="invalid")

        user_ids = torch.randint(0, 1000, (32,))
        movie_ids = torch.randint(0, 5000, (32,))

        with pytest.raises(ValueError, match="Unknown similarity method"):
            model(user_ids, movie_ids)


class TestModelDimensions:
    """Test model with different dimensions."""

    @pytest.mark.parametrize("embedding_dim", [64, 128, 256])
    def test_different_embedding_dims(self, embedding_dim):
        """Test model with different embedding dimensions."""
        model = TwoTowerModel(n_users=1000, n_movies=5000, embedding_dim=embedding_dim)

        user_ids = torch.randint(0, 1000, (32,))
        movie_ids = torch.randint(0, 5000, (32,))

        ratings, user_emb, movie_emb = model(user_ids, movie_ids)

        assert user_emb.shape == (32, embedding_dim)
        assert movie_emb.shape == (32, embedding_dim)

    @pytest.mark.parametrize("batch_size", [1, 16, 64, 128])
    def test_different_batch_sizes(self, batch_size):
        """Test model with different batch sizes."""
        model = TwoTowerModel(n_users=1000, n_movies=5000, embedding_dim=128)

        if batch_size == 1:
            model.eval()  # Batch norm doesn't work with batch_size=1 in train mode

        user_ids = torch.randint(0, 1000, (batch_size,))
        movie_ids = torch.randint(0, 5000, (batch_size,))

        ratings, user_emb, movie_emb = model(user_ids, movie_ids)

        assert ratings.shape == (batch_size,)
        assert user_emb.shape == (batch_size, 128)
        assert movie_emb.shape == (batch_size, 128)
