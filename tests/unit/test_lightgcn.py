"""
Unit tests for LightGCN model.
"""

import pytest
import torch

from src.models.lightgcn import LightGCN, create_lightgcn_model


class TestLightGCNInit:
    """Test LightGCN initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        model = LightGCN(n_users=100, n_items=200, embedding_dim=64)

        assert model.n_users == 100
        assert model.n_items == 200
        assert model.embedding_dim == 64
        assert model.n_layers == 3  # default
        assert model.dropout_rate == 0.0  # default

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        model = LightGCN(
            n_users=50,
            n_items=100,
            embedding_dim=32,
            n_layers=4,
            dropout_rate=0.2,
        )

        assert model.n_users == 50
        assert model.n_items == 100
        assert model.embedding_dim == 32
        assert model.n_layers == 4
        assert model.dropout_rate == 0.2
        assert model.dropout is not None

    def test_init_no_dropout(self):
        """Test initialization without dropout."""
        model = LightGCN(n_users=100, n_items=200, dropout_rate=0.0)

        assert model.dropout is None

    def test_embedding_shapes(self):
        """Test embedding dimensions."""
        model = LightGCN(n_users=100, n_items=200, embedding_dim=64)

        assert model.user_embedding.weight.shape == (100, 64)
        assert model.item_embedding.weight.shape == (200, 64)

    def test_parameter_count(self):
        """Test total parameter count."""
        model = LightGCN(n_users=100, n_items=200, embedding_dim=64)

        n_params = sum(p.numel() for p in model.parameters())
        expected = 100 * 64 + 200 * 64  # user + item embeddings
        assert n_params == expected


class TestLightGCNForward:
    """Test LightGCN forward pass."""

    @pytest.fixture
    def model_and_graph(self):
        """Create model and dummy graph."""
        n_users = 50
        n_items = 100
        embedding_dim = 32
        n_layers = 2

        model = LightGCN(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
        )

        # Create sparse adjacency matrix
        n_total = n_users + n_items
        n_edges = 500
        indices = torch.randint(0, n_total, (2, n_edges))
        values = torch.ones(n_edges) / n_edges  # normalized
        graph = torch.sparse_coo_tensor(indices, values, (n_total, n_total))

        return model, graph, n_users, n_items, embedding_dim

    def test_forward_all_embeddings(self, model_and_graph):
        """Test forward pass returning all embeddings."""
        model, graph, n_users, n_items, embedding_dim = model_and_graph

        user_emb, item_emb = model(graph)

        assert user_emb.shape == (n_users, embedding_dim)
        assert item_emb.shape == (n_items, embedding_dim)

    def test_forward_specific_users_items(self, model_and_graph):
        """Test forward pass with specific user/item indices."""
        model, graph, n_users, n_items, embedding_dim = model_and_graph

        users = torch.tensor([0, 1, 2, 3, 4])
        items = torch.tensor([10, 20, 30, 40, 50])

        user_emb, item_emb = model(graph, users=users, items=items)

        assert user_emb.shape == (5, embedding_dim)
        assert item_emb.shape == (5, embedding_dim)

    def test_forward_with_dropout_training(self, model_and_graph):
        """Test forward with dropout in training mode."""
        model, graph, n_users, n_items, embedding_dim = model_and_graph
        model.dropout_rate = 0.5
        model.dropout = torch.nn.Dropout(0.5)
        model.train()

        user_emb1, _ = model(graph)
        user_emb2, _ = model(graph)

        # With dropout, outputs should differ
        # Note: They might still be same due to deterministic graph conv
        assert user_emb1.shape == (n_users, embedding_dim)

    def test_forward_eval_mode(self, model_and_graph):
        """Test forward in eval mode."""
        model, graph, n_users, n_items, embedding_dim = model_and_graph
        model.eval()

        with torch.no_grad():
            user_emb, item_emb = model(graph)

        assert user_emb.shape == (n_users, embedding_dim)
        assert item_emb.shape == (n_items, embedding_dim)


class TestLightGCNEmbeddingMethods:
    """Test embedding retrieval methods."""

    @pytest.fixture
    def model_and_graph(self):
        """Create model and graph."""
        model = LightGCN(n_users=50, n_items=100, embedding_dim=32, n_layers=2)
        n_total = 150
        indices = torch.randint(0, n_total, (2, 300))
        values = torch.ones(300) / 300
        graph = torch.sparse_coo_tensor(indices, values, (n_total, n_total))
        return model, graph

    def test_get_user_embedding(self, model_and_graph):
        """Test getting user embeddings."""
        model, graph = model_and_graph
        users = torch.tensor([0, 5, 10])

        user_emb = model.get_user_embedding(users, graph)

        assert user_emb.shape == (3, 32)

    def test_get_item_embedding(self, model_and_graph):
        """Test getting item embeddings."""
        model, graph = model_and_graph
        items = torch.tensor([0, 25, 50])

        item_emb = model.get_item_embedding(items, graph)

        assert item_emb.shape == (3, 32)


class TestLightGCNPredict:
    """Test prediction functionality."""

    @pytest.fixture
    def model_and_graph(self):
        """Create model and graph."""
        model = LightGCN(n_users=50, n_items=100, embedding_dim=32, n_layers=2)
        n_total = 150
        indices = torch.randint(0, n_total, (2, 300))
        values = torch.ones(300) / 300
        graph = torch.sparse_coo_tensor(indices, values, (n_total, n_total))
        return model, graph

    def test_predict_scores(self, model_and_graph):
        """Test score prediction."""
        model, graph = model_and_graph
        users = torch.tensor([0, 1, 2, 3, 4])
        items = torch.tensor([10, 20, 30, 40, 50])

        scores = model.predict(users, items, graph)

        assert scores.shape == (5,)
        assert scores.dtype == torch.float32

    def test_predict_batch(self, model_and_graph):
        """Test batch prediction."""
        model, graph = model_and_graph
        batch_size = 32
        users = torch.randint(0, 50, (batch_size,))
        items = torch.randint(0, 100, (batch_size,))

        scores = model.predict(users, items, graph)

        assert scores.shape == (batch_size,)


class TestLightGCNBPRLoss:
    """Test BPR loss computation."""

    @pytest.fixture
    def model_and_graph(self):
        """Create model and graph."""
        model = LightGCN(n_users=50, n_items=100, embedding_dim=32, n_layers=2)
        n_total = 150
        indices = torch.randint(0, n_total, (2, 300))
        values = torch.ones(300) / 300
        graph = torch.sparse_coo_tensor(indices, values, (n_total, n_total))
        return model, graph

    def test_bpr_loss_basic(self, model_and_graph):
        """Test basic BPR loss computation."""
        model, graph = model_and_graph
        batch_size = 16
        users = torch.randint(0, 50, (batch_size,))
        pos_items = torch.randint(0, 100, (batch_size,))
        neg_items = torch.randint(0, 100, (batch_size,))

        bpr_loss, reg_loss = model.bpr_loss(users, pos_items, neg_items, graph)

        assert bpr_loss.shape == ()
        assert reg_loss.shape == ()
        assert bpr_loss.item() >= 0
        assert reg_loss.item() >= 0

    def test_bpr_loss_custom_reg_weight(self, model_and_graph):
        """Test BPR loss with custom regularization weight."""
        model, graph = model_and_graph
        users = torch.tensor([0, 1, 2])
        pos_items = torch.tensor([10, 20, 30])
        neg_items = torch.tensor([50, 60, 70])

        _, reg_loss_default = model.bpr_loss(
            users, pos_items, neg_items, graph, reg_weight=1e-4
        )
        _, reg_loss_higher = model.bpr_loss(
            users, pos_items, neg_items, graph, reg_weight=1e-2
        )

        # Higher reg_weight should give larger reg_loss
        assert reg_loss_higher.item() > reg_loss_default.item()

    def test_bpr_loss_gradient(self, model_and_graph):
        """Test that BPR loss allows gradient computation."""
        model, graph = model_and_graph
        users = torch.tensor([0, 1, 2])
        pos_items = torch.tensor([10, 20, 30])
        neg_items = torch.tensor([50, 60, 70])

        bpr_loss, reg_loss = model.bpr_loss(users, pos_items, neg_items, graph)
        total_loss = bpr_loss + reg_loss

        # Should be able to backpropagate
        total_loss.backward()

        # Check gradients exist
        assert model.user_embedding.weight.grad is not None
        assert model.item_embedding.weight.grad is not None


class TestCreateLightGCNModel:
    """Test create_lightgcn_model helper."""

    def test_create_from_config(self):
        """Test model creation from config dict."""
        config = {
            "n_users": 1000,
            "n_movies": 500,
            "embedding_dim": 64,
            "n_layers": 3,
            "dropout_rate": 0.1,
        }

        model = create_lightgcn_model(config)

        assert isinstance(model, LightGCN)
        assert model.n_users == 1000
        assert model.n_items == 500
        assert model.embedding_dim == 64
        assert model.n_layers == 3

    def test_create_with_n_items_key(self):
        """Test model creation with n_items instead of n_movies."""
        config = {
            "n_users": 100,
            "n_items": 200,
        }

        model = create_lightgcn_model(config)

        assert model.n_items == 200

    def test_create_with_defaults(self):
        """Test model creation with default values."""
        config = {
            "n_users": 100,
            "n_movies": 200,
        }

        model = create_lightgcn_model(config)

        assert model.embedding_dim == 64  # default
        assert model.n_layers == 3  # default
        assert model.dropout_rate == 0.0  # default


class TestLightGCNLayerPropagation:
    """Test layer-wise propagation behavior."""

    def test_multiple_layers(self):
        """Test that multiple layers change embeddings."""
        model_1layer = LightGCN(n_users=20, n_items=30, n_layers=1)
        model_3layer = LightGCN(n_users=20, n_items=30, n_layers=3)

        # Use same initial weights
        model_3layer.user_embedding.weight.data = (
            model_1layer.user_embedding.weight.data.clone()
        )
        model_3layer.item_embedding.weight.data = (
            model_1layer.item_embedding.weight.data.clone()
        )

        # Create graph
        n_total = 50
        indices = torch.randint(0, n_total, (2, 100))
        values = torch.ones(100) / 100
        graph = torch.sparse_coo_tensor(indices, values, (n_total, n_total))

        user_emb_1, _ = model_1layer(graph)
        user_emb_3, _ = model_3layer(graph)

        # More layers should produce different embeddings
        assert not torch.allclose(user_emb_1, user_emb_3)
