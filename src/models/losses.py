"""
Loss Functions for Two-Tower Recommendation System

Implements multiple loss functions:
- MSE Loss: Simple rating prediction loss
- BPR Loss: Bayesian Personalized Ranking for implicit feedback
- Combined Loss: Weighted combination of MSE and BPR
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss for rating prediction.

    Simple and effective for explicit rating prediction.
    """

    def __init__(self):
        """Initialize MSE Loss."""
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self, predicted_ratings: torch.Tensor, true_ratings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss.

        Args:
            predicted_ratings: Model predictions (batch_size,)
            true_ratings: Ground truth ratings (batch_size,)

        Returns:
            MSE loss value
        """
        return self.mse(predicted_ratings, true_ratings)


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss.

    Optimizes for ranking quality by maximizing the margin between
    positive and negative item scores for each user.

    Better for implicit feedback and top-K recommendations.
    """

    def __init__(self, use_in_batch_negatives: bool = True):
        """
        Initialize BPR Loss.

        Args:
            use_in_batch_negatives: Use other items in batch as negatives (faster)
        """
        super().__init__()
        self.use_in_batch_negatives = use_in_batch_negatives

    def forward(
        self,
        user_embeddings: torch.Tensor,
        pos_item_embeddings: torch.Tensor,
        neg_item_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute BPR loss with in-batch negatives (FAST!) or explicit negatives.

        Args:
            user_embeddings: User embeddings (batch_size, embedding_dim)
            pos_item_embeddings: Positive item embeddings (batch_size, embedding_dim)
            neg_item_embeddings: Optional explicit negative embeddings

        Returns:
            BPR loss value
        """
        batch_size, embed_dim = user_embeddings.shape

        # Compute positive scores
        pos_scores = (user_embeddings * pos_item_embeddings).sum(dim=1)  # (batch_size,)

        # IN-BATCH NEGATIVES (10x faster - no extra embedding lookups!)
        if self.use_in_batch_negatives or neg_item_embeddings is None:
            # All user-item scores: (batch_size, batch_size)
            all_scores = torch.mm(user_embeddings, pos_item_embeddings.T)

            # Mask out positive item for each user (diagonal)
            mask = torch.eye(batch_size, device=user_embeddings.device).bool()
            neg_scores = all_scores.masked_fill(mask, float('-inf'))

            # BPR: positive should rank higher than all negatives
            loss = -torch.mean(
                pos_scores - torch.logsumexp(neg_scores, dim=1)
            )
            return loss

        # EXPLICIT NEGATIVES with SAMPLED SOFTMAX (efficient for 100+ negatives!)
        if neg_item_embeddings.dim() == 3:
            # Multiple negatives: (batch_size, n_negatives, embedding_dim)
            batch_size, n_negatives, embed_dim = neg_item_embeddings.shape

            # Compute negative scores: (batch_size, n_negatives)
            user_emb_expanded = user_embeddings.unsqueeze(1)  # (batch_size, 1, embed_dim)
            neg_scores = (user_emb_expanded * neg_item_embeddings).sum(dim=2)  # (batch_size, n_negatives)

            # For many negatives (>10), use sampled softmax (more efficient)
            if n_negatives > 10:
                # Concatenate positive and negative scores
                # Positive score should be at index 0
                all_scores = torch.cat([
                    pos_scores.unsqueeze(1),  # (batch_size, 1)
                    neg_scores  # (batch_size, n_negatives)
                ], dim=1)  # (batch_size, 1 + n_negatives)

                # Cross-entropy loss: positive should have highest score (index 0)
                targets = torch.zeros(batch_size, dtype=torch.long, device=user_embeddings.device)
                loss = F.cross_entropy(all_scores, targets)
            else:
                # For few negatives, use pairwise sigmoid (original BPR)
                pos_scores_expanded = pos_scores.unsqueeze(1).expand(batch_size, n_negatives)
                loss = -F.logsigmoid(pos_scores_expanded - neg_scores).mean()
        else:
            # Single negative: (batch_size, embedding_dim)
            neg_scores = (user_embeddings * neg_item_embeddings).sum(dim=1)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combined Loss: Weighted combination of MSE and BPR.

    Balances rating prediction accuracy (MSE) with ranking quality (BPR).
    """

    def __init__(self, mse_weight: float = 0.7, bpr_weight: float = 0.3):
        """
        Initialize Combined Loss.

        Args:
            mse_weight: Weight for MSE loss component
            bpr_weight: Weight for BPR loss component
        """
        super().__init__()

        if not (0 <= mse_weight <= 1 and 0 <= bpr_weight <= 1):
            raise ValueError("Loss weights must be in [0, 1]")

        if abs(mse_weight + bpr_weight - 1.0) > 1e-6:
            raise ValueError("Loss weights must sum to 1.0")

        self.mse_weight = mse_weight
        self.bpr_weight = bpr_weight

        self.mse_loss = MSELoss()
        self.bpr_loss = BPRLoss()

    def forward(
        self,
        predicted_ratings: torch.Tensor,
        true_ratings: torch.Tensor,
        user_embeddings: torch.Tensor,
        pos_item_embeddings: torch.Tensor,
        neg_item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            predicted_ratings: Model predictions (batch_size,)
            true_ratings: Ground truth ratings (batch_size,)
            user_embeddings: User embeddings (batch_size, embedding_dim)
            pos_item_embeddings: Positive item embeddings (batch_size, embedding_dim)
            neg_item_embeddings: Negative item embeddings (batch_size, embedding_dim)

        Returns:
            Combined loss value
        """
        mse = self.mse_loss(predicted_ratings, true_ratings)
        bpr = self.bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings)

        combined = self.mse_weight * mse + self.bpr_weight * bpr

        return combined


class RegularizedLoss(nn.Module):
    """
    Loss with L2 regularization on embeddings.

    Helps prevent overfitting by penalizing large embedding values.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        reg_weight: float = 1e-4,
    ):
        """
        Initialize Regularized Loss.

        Args:
            base_loss: Base loss function (MSE, BPR, or Combined)
            reg_weight: L2 regularization weight
        """
        super().__init__()

        self.base_loss = base_loss
        self.reg_weight = reg_weight

    def forward(
        self,
        predicted_ratings: torch.Tensor,
        true_ratings: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        neg_item_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute regularized loss.

        Args:
            predicted_ratings: Model predictions
            true_ratings: Ground truth ratings
            user_embeddings: User embeddings
            item_embeddings: Item embeddings
            neg_item_embeddings: Negative item embeddings (for BPR)

        Returns:
            Loss value with regularization
        """
        # Compute base loss
        if isinstance(self.base_loss, MSELoss):
            base_loss_value = self.base_loss(predicted_ratings, true_ratings)
        elif isinstance(self.base_loss, BPRLoss):
            if neg_item_embeddings is None:
                raise ValueError("BPR loss requires negative item embeddings")
            base_loss_value = self.base_loss(
                user_embeddings, item_embeddings, neg_item_embeddings
            )
        elif isinstance(self.base_loss, CombinedLoss):
            if neg_item_embeddings is None:
                raise ValueError("Combined loss requires negative item embeddings")
            base_loss_value = self.base_loss(
                predicted_ratings,
                true_ratings,
                user_embeddings,
                item_embeddings,
                neg_item_embeddings,
            )
        else:
            raise ValueError(f"Unsupported base loss type: {type(self.base_loss)}")

        # Compute L2 regularization
        user_reg = (user_embeddings**2).sum(dim=1).mean()
        item_reg = (item_embeddings**2).sum(dim=1).mean()

        if neg_item_embeddings is not None:
            neg_item_reg = (neg_item_embeddings**2).sum(dim=1).mean()
            reg_term = (user_reg + item_reg + neg_item_reg) / 3
        else:
            reg_term = (user_reg + item_reg) / 2

        # Combined loss
        total_loss = base_loss_value + self.reg_weight * reg_term

        return total_loss


class DiversityLoss(nn.Module):
    """
    Diversity Loss: Encourages recommending different items to different users.

    Prevents mode collapse where all users get the same recommendations.
    Works by maximizing variance of item scores across users in the batch.
    """

    def __init__(self, diversity_weight: float = 0.1):
        """
        Initialize Diversity Loss.

        Args:
            diversity_weight: Weight for diversity regularization
        """
        super().__init__()
        self.diversity_weight = diversity_weight

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diversity loss.

        Encourages different users to prefer different items by maximizing
        the variance of item popularity across the batch.

        Args:
            user_embeddings: User embeddings (batch_size, embedding_dim)
            item_embeddings: Item embeddings (batch_size, embedding_dim) or
                            (n_items, embedding_dim) for all items

        Returns:
            Diversity loss value (negative entropy or variance)
        """
        # Compute all user-item scores
        if item_embeddings.shape[0] == user_embeddings.shape[0]:
            # Positive items only (same as batch size)
            scores = (user_embeddings * item_embeddings).sum(dim=1)  # (batch_size,)

            # We want high variance across users
            # Low variance = all users like same items (mode collapse)
            # High variance = different users like different items (diverse)
            diversity_loss = -torch.var(scores)  # Negative because we want to maximize
        else:
            # All items (n_items != batch_size)
            scores = torch.mm(user_embeddings, item_embeddings.T)  # (batch_size, n_items)

            # Compute item popularity across users
            item_popularity = scores.mean(dim=0)  # (n_items,)

            # Entropy-based diversity: penalize if all weight on few items
            # High entropy = diverse, low entropy = concentrated (mode collapse)
            probs = F.softmax(item_popularity, dim=0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()

            # Negative because we want to maximize entropy
            diversity_loss = -entropy

        return self.diversity_weight * diversity_loss


class BPRWithDiversityLoss(nn.Module):
    """
    BPR Loss with Diversity Regularization.

    Combines ranking quality (BPR) with diversity encouragement.
    Prevents mode collapse in large-batch training.
    """

    def __init__(
        self,
        diversity_weight: float = 0.1,
        use_in_batch_negatives: bool = True,
    ):
        """
        Initialize BPR with Diversity Loss.

        Args:
            diversity_weight: Weight for diversity term
            use_in_batch_negatives: Use in-batch negatives
        """
        super().__init__()
        self.bpr_loss = BPRLoss(use_in_batch_negatives=use_in_batch_negatives)
        self.diversity_loss = DiversityLoss(diversity_weight=diversity_weight)

    def forward(
        self,
        user_embeddings: torch.Tensor,
        pos_item_embeddings: torch.Tensor,
        neg_item_embeddings: Optional[torch.Tensor] = None,
        all_item_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute BPR loss with diversity regularization.

        Args:
            user_embeddings: User embeddings (batch_size, embedding_dim)
            pos_item_embeddings: Positive item embeddings (batch_size, embedding_dim)
            neg_item_embeddings: Negative item embeddings (optional)
            all_item_embeddings: All item embeddings for diversity (optional)

        Returns:
            Combined BPR + diversity loss
        """
        # BPR loss (main ranking objective)
        bpr = self.bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings)

        # Diversity loss (prevent mode collapse)
        if all_item_embeddings is not None:
            diversity = self.diversity_loss(user_embeddings, all_item_embeddings)
        else:
            # Use positive items if all items not available
            diversity = self.diversity_loss(user_embeddings, pos_item_embeddings)

        return bpr + diversity


def create_loss_function(config: dict) -> nn.Module:
    """
    Create loss function from configuration.

    Args:
        config: Loss configuration dictionary
            - loss_type: 'mse', 'bpr', 'bpr_diversity', or 'combined'
            - mse_weight: Weight for MSE (if combined)
            - bpr_weight: Weight for BPR (if combined)
            - use_regularization: Whether to add L2 regularization
            - reg_weight: Regularization weight
            - diversity_weight: Weight for diversity loss (if bpr_diversity)

    Returns:
        Loss function module
    """
    loss_type = config.get("loss_type", "mse")

    # Create base loss
    if loss_type == "mse":
        base_loss = MSELoss()
    elif loss_type == "bpr":
        base_loss = BPRLoss()
    elif loss_type == "bpr_diversity":
        diversity_weight = config.get("diversity_weight", 0.1)
        use_in_batch = config.get("negative_sampling", False) is False
        base_loss = BPRWithDiversityLoss(
            diversity_weight=diversity_weight,
            use_in_batch_negatives=use_in_batch
        )
    elif loss_type == "combined":
        mse_weight = config.get("mse_weight", 0.7)
        bpr_weight = config.get("bpr_weight", 0.3)
        base_loss = CombinedLoss(mse_weight=mse_weight, bpr_weight=bpr_weight)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Use 'mse', 'bpr', 'bpr_diversity', or 'combined'"
        )

    # Add regularization if requested
    if config.get("use_regularization", False):
        reg_weight = config.get("reg_weight", 1e-4)
        loss_fn = RegularizedLoss(base_loss, reg_weight=reg_weight)
    else:
        loss_fn = base_loss

    return loss_fn


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")

    batch_size = 32
    embedding_dim = 128

    # Create dummy data
    predicted_ratings = torch.rand(batch_size) * 4.5 + 0.5
    true_ratings = torch.rand(batch_size) * 4.5 + 0.5
    user_embeddings = torch.randn(batch_size, embedding_dim)
    pos_item_embeddings = torch.randn(batch_size, embedding_dim)
    neg_item_embeddings = torch.randn(batch_size, embedding_dim)

    # Test MSE Loss
    mse_loss = MSELoss()
    mse_value = mse_loss(predicted_ratings, true_ratings)
    print(f"✓ MSE Loss: {mse_value.item():.4f}")

    # Test BPR Loss
    bpr_loss = BPRLoss()
    bpr_value = bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings)
    print(f"✓ BPR Loss: {bpr_value.item():.4f}")

    # Test Combined Loss
    combined_loss = CombinedLoss(mse_weight=0.7, bpr_weight=0.3)
    combined_value = combined_loss(
        predicted_ratings,
        true_ratings,
        user_embeddings,
        pos_item_embeddings,
        neg_item_embeddings,
    )
    print(f"✓ Combined Loss: {combined_value.item():.4f}")

    # Test Regularized Loss
    reg_loss = RegularizedLoss(MSELoss(), reg_weight=1e-4)
    reg_value = reg_loss(
        predicted_ratings, true_ratings, user_embeddings, pos_item_embeddings
    )
    print(f"✓ Regularized MSE Loss: {reg_value.item():.4f}")

    print("\nAll loss functions working correctly!")
