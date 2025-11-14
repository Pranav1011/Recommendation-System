"""
Hybrid LightGCN: Graph Structure + Side Features

Combines:
- LightGCN for collaborative filtering (graph structure)
- Feature MLPs for content-based filtering (genres, metadata)

Best of both worlds for sparse recommendation datasets.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureMLP(nn.Module):
    """MLP for processing user/movie features."""

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize feature MLP.

        Args:
            feature_dim: Input feature dimension
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

        if feature_dim > 0:
            # Feature transformation network
            self.network = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
        else:
            self.network = None

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        if self.network is not None:
            for module in self.network.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Input features (batch_size, feature_dim)

        Returns:
            Feature embeddings (batch_size, embedding_dim)
        """
        if self.network is None or self.feature_dim == 0:
            # No features, return zeros
            batch_size = features.size(0) if features.numel() > 0 else 1
            return torch.zeros(batch_size, self.embedding_dim, device=features.device)

        return self.network(features)


class HybridLightGCN(nn.Module):
    """
    Hybrid model combining LightGCN with feature information.

    Architecture:
        User representation = Graph embedding + Feature embedding
        Item representation = Graph embedding + Feature embedding
        Score = dot product of representations
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 128,
        n_layers: int = 3,
        feature_hidden_dim: int = 128,
        dropout_rate: float = 0.2,
        feature_weight: float = 0.3,
    ):
        """
        Initialize Hybrid LightGCN.

        Args:
            n_users: Number of users
            n_items: Number of items
            user_feature_dim: User feature dimension
            item_feature_dim: Item feature dimension
            embedding_dim: Embedding dimension for both graph and features
            n_layers: Number of LightGCN layers
            feature_hidden_dim: Hidden dimension for feature MLPs
            dropout_rate: Dropout rate
            feature_weight: Weight for feature embeddings (0-1)
                           Final = (1-w)*graph + w*features
        """
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.feature_weight = feature_weight

        logger.info("Initializing Hybrid LightGCN:")
        logger.info(f"  Users: {n_users:,}")
        logger.info(f"  Items: {n_items:,}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  LightGCN layers: {n_layers}")
        logger.info(f"  User feature dim: {user_feature_dim}")
        logger.info(f"  Item feature dim: {item_feature_dim}")
        logger.info(f"  Feature weight: {feature_weight}")

        # ========== Graph Component (LightGCN) ==========
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # ========== Feature Component (MLPs) ==========
        self.user_feature_mlp = FeatureMLP(
            feature_dim=user_feature_dim,
            embedding_dim=embedding_dim,
            hidden_dim=feature_hidden_dim,
            dropout_rate=dropout_rate,
        )

        self.item_feature_mlp = FeatureMLP(
            feature_dim=item_feature_dim,
            embedding_dim=embedding_dim,
            hidden_dim=feature_hidden_dim,
            dropout_rate=dropout_rate,
        )

        # Initialize embeddings
        self._init_weights()

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  Total parameters: {n_params:,}")

    def _init_weights(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def lightgcn_forward(
        self,
        graph: torch.sparse.FloatTensor,
        users: Optional[torch.Tensor] = None,
        items: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN graph convolution.

        Args:
            graph: Normalized adjacency matrix
            users: Optional user indices
            items: Optional item indices

        Returns:
            (user_graph_embeddings, item_graph_embeddings)
        """
        # Get initial embeddings
        all_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )

        # Store embeddings at each layer
        embeddings_layers = [all_embeddings]

        # Graph convolution
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(graph, all_embeddings)
            embeddings_layers.append(all_embeddings)

        # Aggregate embeddings from all layers (mean)
        final_embeddings = torch.mean(torch.stack(embeddings_layers, dim=0), dim=0)

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            final_embeddings, [self.n_users, self.n_items], dim=0
        )

        # If specific users/items requested
        if users is not None and items is not None:
            user_emb = user_all_embeddings[users]
            item_emb = item_all_embeddings[items]
            return user_emb, item_emb

        return user_all_embeddings, item_all_embeddings

    def forward(
        self,
        graph: torch.sparse.FloatTensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
        users: Optional[torch.Tensor] = None,
        items: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining graph and features.

        Args:
            graph: Normalized adjacency matrix
            user_features: User features (n_users, user_feat_dim) or batch subset
            item_features: Item features (n_items, item_feat_dim) or batch subset
            users: Optional user indices for batch
            items: Optional item indices for batch

        Returns:
            (user_final_embeddings, item_final_embeddings)
        """
        # Get graph embeddings
        user_graph_emb, item_graph_emb = self.lightgcn_forward(graph, users, items)

        # Get feature embeddings
        if user_features is not None:
            user_feat_emb = self.user_feature_mlp(user_features)
        else:
            user_feat_emb = torch.zeros_like(user_graph_emb)

        if item_features is not None:
            item_feat_emb = self.item_feature_mlp(item_features)
        else:
            item_feat_emb = torch.zeros_like(item_graph_emb)

        # Combine graph and feature embeddings
        # Weighted combination: (1-w)*graph + w*features
        user_final = (
            1 - self.feature_weight
        ) * user_graph_emb + self.feature_weight * user_feat_emb
        item_final = (
            1 - self.feature_weight
        ) * item_graph_emb + self.feature_weight * item_feat_emb

        # L2 normalize
        user_final = F.normalize(user_final, p=2, dim=1)
        item_final = F.normalize(item_final, p=2, dim=1)

        return user_final, item_final

    def predict(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        graph: torch.sparse.FloatTensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict scores for user-item pairs.

        Args:
            users: User indices (batch_size,)
            items: Item indices (batch_size,)
            graph: Normalized adjacency matrix
            user_features: User features for batch
            item_features: Item features for batch

        Returns:
            Predicted scores (batch_size,)
        """
        user_emb, item_emb = self.forward(
            graph, user_features, item_features, users, items
        )

        # Inner product
        scores = (user_emb * item_emb).sum(dim=1)

        return scores

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        graph: torch.sparse.FloatTensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
        reg_weight: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute BPR loss for training.

        Args:
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative item indices
            graph: Normalized adjacency matrix
            user_features: Full user feature matrix (n_users, user_feat_dim)
            item_features: Full item feature matrix (n_items, item_feat_dim)
            reg_weight: L2 regularization weight

        Returns:
            (bpr_loss, reg_loss)
        """
        # Get all embeddings with features
        user_all_emb, item_all_emb = self.forward(graph, user_features, item_features)

        # Get embeddings for batch
        user_emb = user_all_emb[users]
        pos_emb = item_all_emb[pos_items]
        neg_emb = item_all_emb[neg_items]

        # Positive and negative scores
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)

        # BPR loss
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 regularization on initial embeddings (graph part only)
        user_emb_0 = self.user_embedding(users)
        pos_emb_0 = self.item_embedding(pos_items)
        neg_emb_0 = self.item_embedding(neg_items)

        reg_loss = (
            (user_emb_0**2).sum() + (pos_emb_0**2).sum() + (neg_emb_0**2).sum()
        ) / (2 * users.size(0))

        return bpr_loss, reg_weight * reg_loss


def create_hybrid_lightgcn_model(config: dict) -> HybridLightGCN:
    """
    Create Hybrid LightGCN model from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        HybridLightGCN model
    """
    return HybridLightGCN(
        n_users=config["n_users"],
        n_items=config.get("n_movies", config.get("n_items")),
        user_feature_dim=config.get("user_feature_dim", 0),
        item_feature_dim=config.get(
            "movie_feature_dim", config.get("item_feature_dim", 0)
        ),
        embedding_dim=config.get("embedding_dim", 128),
        n_layers=config.get("n_layers", 3),
        feature_hidden_dim=config.get("feature_hidden_dim", 128),
        dropout_rate=config.get("dropout_rate", 0.2),
        feature_weight=config.get("feature_weight", 0.3),
    )


if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)

    print("Testing Hybrid LightGCN Model...")

    # Model config
    config = {
        "n_users": 162541,
        "n_movies": 59047,
        "user_feature_dim": 30,
        "movie_feature_dim": 13,
        "embedding_dim": 128,
        "n_layers": 3,
        "feature_hidden_dim": 128,
        "dropout_rate": 0.2,
        "feature_weight": 0.3,
    }

    # Create model
    model = create_hybrid_lightgcn_model(config)

    # Create dummy inputs
    batch_size = 32
    users = torch.randint(0, config["n_users"], (batch_size,))
    pos_items = torch.randint(0, config["n_movies"], (batch_size,))
    neg_items = torch.randint(0, config["n_movies"], (batch_size,))

    user_features = torch.randn(batch_size, config["user_feature_dim"])
    item_features_pos = torch.randn(batch_size, config["movie_feature_dim"])
    item_features_neg = torch.randn(batch_size, config["movie_feature_dim"])

    # Create dummy graph
    n_total = config["n_users"] + config["n_movies"]
    indices = torch.randint(0, n_total, (2, 10000))
    values = torch.rand(10000)
    graph = torch.sparse.FloatTensor(indices, values, torch.Size([n_total, n_total]))

    print(
        f"\n✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Test forward pass
    user_emb, item_emb = model(graph)
    print("\n✓ Forward pass:")
    print(f"  User embeddings: {user_emb.shape}")
    print(f"  Item embeddings: {item_emb.shape}")

    # Test BPR loss
    bpr_loss, reg_loss = model.bpr_loss(users, pos_items, neg_items, graph)
    print("\n✓ BPR Loss:")
    print(f"  BPR loss: {bpr_loss.item():.4f}")
    print(f"  Reg loss: {reg_loss.item():.6f}")

    print("\nHybrid LightGCN model working correctly!")
