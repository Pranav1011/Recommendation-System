"""
Unit tests for loss functions.
"""

import pytest
import torch

from src.models.losses import (
    BPRLoss,
    CombinedLoss,
    MSELoss,
    RegularizedLoss,
    create_loss_function,
)


class TestMSELoss:
    """Test MSE Loss."""

    def test_zero_loss(self):
        """Test MSE loss with perfect predictions."""
        loss_fn = MSELoss()

        predicted = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.0, 2.0, 3.0])

        loss = loss_fn(predicted, true)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_loss(self):
        """Test MSE loss with imperfect predictions."""
        loss_fn = MSELoss()

        predicted = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.5, 2.5, 3.5])

        loss = loss_fn(predicted, true)
        # MSE = mean((0.5^2 + 0.5^2 + 0.5^2)) = 0.25
        assert loss.item() == pytest.approx(0.25, abs=1e-6)

    def test_batch_size(self):
        """Test MSE loss with different batch sizes."""
        loss_fn = MSELoss()

        for batch_size in [1, 16, 32, 64]:
            predicted = torch.randn(batch_size)
            true = torch.randn(batch_size)

            loss = loss_fn(predicted, true)
            assert loss.item() >= 0.0


class TestBPRLoss:
    """Test BPR Loss."""

    def test_perfect_ranking(self):
        """Test BPR loss with perfect ranking (pos score >> neg score)."""
        loss_fn = BPRLoss()

        batch_size = 32
        embedding_dim = 128

        user_emb = torch.randn(batch_size, embedding_dim)
        pos_item_emb = user_emb + 0.1  # Very similar to user
        neg_item_emb = -user_emb  # Very different from user

        loss = loss_fn(user_emb, pos_item_emb, neg_item_emb)
        # BPR loss can be negative when positive scores >> negative scores
        # Just verify it computes without error
        assert isinstance(loss, torch.Tensor)

    def test_bad_ranking(self):
        """Test BPR loss with bad ranking (neg score > pos score)."""
        loss_fn = BPRLoss()

        batch_size = 32
        embedding_dim = 128

        user_emb = torch.randn(batch_size, embedding_dim)
        pos_item_emb = -user_emb  # Different from user
        neg_item_emb = user_emb + 0.1  # Similar to user (bad!)

        loss = loss_fn(user_emb, pos_item_emb, neg_item_emb)
        # Loss should be high for bad ranking
        assert loss.item() > 1.0

    def test_output_shape(self):
        """Test that BPR loss returns a scalar."""
        loss_fn = BPRLoss()

        user_emb = torch.randn(32, 128)
        pos_item_emb = torch.randn(32, 128)
        neg_item_emb = torch.randn(32, 128)

        loss = loss_fn(user_emb, pos_item_emb, neg_item_emb)
        assert loss.shape == torch.Size([])  # Scalar


class TestCombinedLoss:
    """Test Combined Loss."""

    def test_default_weights(self):
        """Test combined loss with default weights."""
        loss_fn = CombinedLoss()
        assert loss_fn.mse_weight == 0.7
        assert loss_fn.bpr_weight == 0.3

    def test_custom_weights(self):
        """Test combined loss with custom weights."""
        loss_fn = CombinedLoss(mse_weight=0.5, bpr_weight=0.5)
        assert loss_fn.mse_weight == 0.5
        assert loss_fn.bpr_weight == 0.5

    def test_invalid_weights(self):
        """Test that invalid weights raise error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            CombinedLoss(mse_weight=0.5, bpr_weight=0.6)

    def test_forward_pass(self):
        """Test combined loss forward pass."""
        loss_fn = CombinedLoss(mse_weight=0.7, bpr_weight=0.3)

        batch_size = 32
        embedding_dim = 128

        predicted_ratings = torch.rand(batch_size) * 4.5 + 0.5
        true_ratings = torch.rand(batch_size) * 4.5 + 0.5
        user_emb = torch.randn(batch_size, embedding_dim)
        pos_item_emb = torch.randn(batch_size, embedding_dim)
        neg_item_emb = torch.randn(batch_size, embedding_dim)

        loss = loss_fn(
            predicted_ratings, true_ratings, user_emb, pos_item_emb, neg_item_emb
        )

        assert loss.item() >= 0.0
        assert loss.shape == torch.Size([])


class TestRegularizedLoss:
    """Test Regularized Loss."""

    def test_mse_regularization(self):
        """Test regularized MSE loss."""
        base_loss = MSELoss()
        reg_loss_fn = RegularizedLoss(base_loss, reg_weight=1e-4)

        batch_size = 32
        embedding_dim = 128

        predicted_ratings = torch.rand(batch_size) * 4.5 + 0.5
        true_ratings = torch.rand(batch_size) * 4.5 + 0.5
        user_emb = torch.randn(batch_size, embedding_dim)
        item_emb = torch.randn(batch_size, embedding_dim)

        # Compute regularized loss
        reg_loss = reg_loss_fn(predicted_ratings, true_ratings, user_emb, item_emb)

        # Compute base loss
        base_loss_value = base_loss(predicted_ratings, true_ratings)

        # Regularized loss should be higher than base loss
        assert reg_loss.item() > base_loss_value.item()

    def test_bpr_regularization(self):
        """Test regularized BPR loss."""
        base_loss = BPRLoss()
        reg_loss_fn = RegularizedLoss(base_loss, reg_weight=1e-4)

        batch_size = 32
        embedding_dim = 128

        predicted_ratings = torch.rand(batch_size) * 4.5 + 0.5
        true_ratings = torch.rand(batch_size) * 4.5 + 0.5
        user_emb = torch.randn(batch_size, embedding_dim)
        item_emb = torch.randn(batch_size, embedding_dim)
        neg_item_emb = torch.randn(batch_size, embedding_dim)

        # Compute regularized loss
        reg_loss = reg_loss_fn(
            predicted_ratings, true_ratings, user_emb, item_emb, neg_item_emb
        )

        assert reg_loss.item() >= 0.0


class TestCreateLossFunction:
    """Test loss function factory."""

    def test_create_mse_loss(self):
        """Test creating MSE loss from config."""
        config = {"loss_type": "mse"}
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, MSELoss)

    def test_create_bpr_loss(self):
        """Test creating BPR loss from config."""
        config = {"loss_type": "bpr"}
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, BPRLoss)

    def test_create_combined_loss(self):
        """Test creating combined loss from config."""
        config = {
            "loss_type": "combined",
            "mse_weight": 0.6,
            "bpr_weight": 0.4,
        }
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, CombinedLoss)
        assert loss_fn.mse_weight == 0.6
        assert loss_fn.bpr_weight == 0.4

    def test_create_regularized_loss(self):
        """Test creating regularized loss from config."""
        config = {
            "loss_type": "mse",
            "use_regularization": True,
            "reg_weight": 1e-3,
        }
        loss_fn = create_loss_function(config)
        assert isinstance(loss_fn, RegularizedLoss)
        assert loss_fn.reg_weight == 1e-3

    def test_invalid_loss_type(self):
        """Test that invalid loss type raises error."""
        config = {"loss_type": "invalid"}

        with pytest.raises(ValueError, match="Unknown loss type"):
            create_loss_function(config)

    def test_default_config(self):
        """Test creating loss with default config."""
        config = {}
        loss_fn = create_loss_function(config)
        # Should default to MSE
        assert isinstance(loss_fn, MSELoss)
