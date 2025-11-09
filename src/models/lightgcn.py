"""
LightGCN: Simplified Graph Convolution Network for Recommendation

Paper: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
       (He et al., SIGIR 2020)

Key ideas:
- Remove feature transformation and nonlinear activation
- Only neighborhood aggregation matters
- Simple weighted sum of embeddings across layers
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LightGCN(nn.Module):
    """
    LightGCN Recommendation Model.

    Simplified GCN that removes unnecessary complexity:
    - No feature transformation matrices
    - No nonlinear activations
    - Only neighborhood aggregation (averaging)

    Achieves state-of-the-art performance with fewer parameters.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize LightGCN.

        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Embedding dimension (paper uses 64)
            n_layers: Number of graph convolution layers (paper uses 3)
            dropout_rate: Dropout rate (optional, paper doesn't use it)
        """
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        logger.info(f"Initializing LightGCN:")
        logger.info(f"  Users: {n_users:,}")
        logger.info(f"  Items: {n_items:,}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Layers: {n_layers}")

        # User and item embeddings (learnable)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Dropout (optional)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        # Initialize embeddings
        self._init_weights()

        # Total parameters
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  Total parameters: {n_params:,}")

    def _init_weights(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(
        self,
        graph: torch.sparse.FloatTensor,
        users: Optional[torch.Tensor] = None,
        items: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LightGCN.

        Args:
            graph: Normalized adjacency matrix (sparse)
                   Shape: (n_users + n_items, n_users + n_items)
            users: Optional user indices for prediction
            items: Optional item indices for prediction

        Returns:
            Tuple of (user_embeddings, item_embeddings) or (user_final, item_final)
            depending on whether user/item indices are provided
        """
        # Get initial embeddings
        all_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        # Shape: (n_users + n_items, embedding_dim)

        # Store embeddings at each layer
        embeddings_layers = [all_embeddings]

        # Graph convolution (light propagation)
        for layer in range(self.n_layers):
            # E^(k+1) = A * E^(k)
            # A is normalized adjacency, E^(k) is embeddings at layer k
            all_embeddings = torch.sparse.mm(graph, all_embeddings)

            # Optional dropout (paper doesn't use it)
            if self.dropout is not None and self.training:
                all_embeddings = self.dropout(all_embeddings)

            embeddings_layers.append(all_embeddings)

        # Aggregate embeddings from all layers (including layer 0)
        # Final embedding is mean of all layers
        final_embeddings = torch.mean(torch.stack(embeddings_layers, dim=0), dim=0)
        # Shape: (n_users + n_items, embedding_dim)

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            final_embeddings, [self.n_users, self.n_items], dim=0
        )

        # If specific users/items requested, return their embeddings
        if users is not None and items is not None:
            user_final = user_all_embeddings[users]
            item_final = item_all_embeddings[items]
            return user_final, item_final

        # Otherwise return all embeddings
        return user_all_embeddings, item_all_embeddings

    def get_user_embedding(self, users: torch.Tensor, graph: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Get embeddings for specific users.

        Args:
            users: User indices (batch_size,)
            graph: Normalized adjacency matrix

        Returns:
            User embeddings (batch_size, embedding_dim)
        """
        user_embeddings, _ = self.forward(graph)
        return user_embeddings[users]

    def get_item_embedding(self, items: torch.Tensor, graph: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Get embeddings for specific items.

        Args:
            items: Item indices (batch_size,)
            graph: Normalized adjacency matrix

        Returns:
            Item embeddings (batch_size, embedding_dim)
        """
        _, item_embeddings = self.forward(graph)
        return item_embeddings[items]

    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        graph: torch.sparse.FloatTensor,
    ) -> torch.Tensor:
        """
        Predict scores for user-item pairs.

        Args:
            users: User indices (batch_size,)
            items: Item indices (batch_size,)
            graph: Normalized adjacency matrix

        Returns:
            Predicted scores (batch_size,)
        """
        user_emb, item_emb = self.forward(graph, users, items)

        # Inner product
        scores = (user_emb * item_emb).sum(dim=1)

        return scores

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        graph: torch.sparse.FloatTensor,
        reg_weight: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute BPR loss for training.

        Args:
            users: User indices (batch_size,)
            pos_items: Positive item indices (batch_size,)
            neg_items: Negative item indices (batch_size,)
            graph: Normalized adjacency matrix
            reg_weight: L2 regularization weight

        Returns:
            Tuple of (bpr_loss, reg_loss)
        """
        # Get all embeddings
        user_all_emb, item_all_emb = self.forward(graph)

        # Get embeddings for batch
        user_emb = user_all_emb[users]  # (batch_size, embedding_dim)
        pos_emb = item_all_emb[pos_items]  # (batch_size, embedding_dim)
        neg_emb = item_all_emb[neg_items]  # (batch_size, embedding_dim)

        # Positive scores
        pos_scores = (user_emb * pos_emb).sum(dim=1)  # (batch_size,)

        # Negative scores
        neg_scores = (user_emb * neg_emb).sum(dim=1)  # (batch_size,)

        # BPR loss: -log(sigmoid(pos_score - neg_score))
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 regularization on initial embeddings (not propagated ones)
        # This is important to prevent overfitting
        user_emb_0 = self.user_embedding(users)
        pos_emb_0 = self.item_embedding(pos_items)
        neg_emb_0 = self.item_embedding(neg_items)

        reg_loss = (
            (user_emb_0**2).sum()
            + (pos_emb_0**2).sum()
            + (neg_emb_0**2).sum()
        ) / (2 * users.size(0))

        return bpr_loss, reg_weight * reg_loss


def create_lightgcn_model(config: dict) -> LightGCN:
    """
    Create LightGCN model from configuration.

    Args:
        config: Model configuration dictionary
            - n_users: Number of users
            - n_movies/n_items: Number of items
            - embedding_dim: Embedding dimension (default: 64)
            - n_layers: Number of GCN layers (default: 3)
            - dropout_rate: Dropout rate (default: 0.0)

    Returns:
        LightGCN model
    """
    return LightGCN(
        n_users=config["n_users"],
        n_items=config.get("n_movies", config.get("n_items")),
        embedding_dim=config.get("embedding_dim", 64),
        n_layers=config.get("n_layers", 3),
        dropout_rate=config.get("dropout_rate", 0.0),
    )


if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)

    print("Testing LightGCN Model...")

    # Model config
    config = {
        "n_users": 162541,
        "n_movies": 59047,
        "embedding_dim": 64,
        "n_layers": 3,
    }

    # Create model
    model = create_lightgcn_model(config)

    # Create dummy graph (random sparse matrix for testing)
    n_total = config["n_users"] + config["n_movies"]
    indices = torch.randint(0, n_total, (2, 10000))  # 10K edges
    values = torch.rand(10000)
    size = torch.Size([n_total, n_total])
    graph = torch.sparse.FloatTensor(indices, values, size)

    # Test forward pass
    user_all_emb, item_all_emb = model(graph)
    print(f"\n✓ Forward pass:")
    print(f"  User embeddings: {user_all_emb.shape}")
    print(f"  Item embeddings: {item_all_emb.shape}")

    # Test prediction
    batch_size = 32
    users = torch.randint(0, config["n_users"], (batch_size,))
    pos_items = torch.randint(0, config["n_movies"], (batch_size,))
    neg_items = torch.randint(0, config["n_movies"], (batch_size,))

    scores = model.predict(users, pos_items, graph)
    print(f"\n✓ Prediction:")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")

    # Test BPR loss
    bpr_loss, reg_loss = model.bpr_loss(users, pos_items, neg_items, graph)
    print(f"\n✓ BPR Loss:")
    print(f"  BPR loss: {bpr_loss.item():.4f}")
    print(f"  Reg loss: {reg_loss.item():.6f}")

    print("\nLightGCN model working correctly!")
