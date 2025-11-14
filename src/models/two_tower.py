"""
Two-Tower Neural Collaborative Filtering Model

Implements a Two-Tower architecture for recommendation systems:
- User Tower: Learns user representations from user ID and features
- Movie Tower: Learns movie representations from movie ID and features
- Prediction: Cosine similarity or dot product of embeddings
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """
    User Tower: Learns user embeddings from user ID and features.

    Architecture:
    - User ID Embedding
    - User Feature Transform (optional)
    - Dense layers with batch norm and dropout
    - Output: User embedding vector
    """

    def __init__(
        self,
        n_users: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        feature_dim: int = 30,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize User Tower.

        Args:
            n_users: Number of unique users
            embedding_dim: Dimension of user embedding
            hidden_dim: Hidden layer dimension
            feature_dim: Number of user features
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.n_users = n_users
        self.embedding_dim = embedding_dim

        # User ID embedding
        self.user_embedding = nn.Embedding(n_users, embedding_dim)

        # Feature transformation (if features provided)
        self.feature_transform = nn.Linear(feature_dim, 32)

        # Dense layers
        input_dim = embedding_dim + 32
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.feature_transform.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        # Initialize biases to zero
        nn.init.zeros_(self.feature_transform.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self, user_ids: torch.Tensor, user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through user tower.

        Args:
            user_ids: User IDs (batch_size,)
            user_features: User features (batch_size, feature_dim)

        Returns:
            User embeddings (batch_size, embedding_dim)
        """
        # Get user ID embedding
        user_emb = self.user_embedding(user_ids)

        # Transform user features
        if user_features is not None:
            features = F.relu(self.feature_transform(user_features))
            # Concatenate ID embedding and features
            x = torch.cat([user_emb, features], dim=1)
        else:
            # Use only ID embedding (pad to expected dimension)
            padding = torch.zeros(
                user_emb.size(0), 32, device=user_emb.device, dtype=user_emb.dtype
            )
            x = torch.cat([user_emb, padding], dim=1)

        # Dense layers with batch norm and dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # L2 normalization for cosine similarity
        x = F.normalize(x, p=2, dim=1)

        return x


class MovieTower(nn.Module):
    """
    Movie Tower: Learns movie embeddings from movie ID and features.

    Architecture:
    - Movie ID Embedding
    - Movie Feature Transform (optional)
    - Dense layers with batch norm and dropout
    - Output: Movie embedding vector
    """

    def __init__(
        self,
        n_movies: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        feature_dim: int = 12,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize Movie Tower.

        Args:
            n_movies: Number of unique movies
            embedding_dim: Dimension of movie embedding
            hidden_dim: Hidden layer dimension
            feature_dim: Number of movie features
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.n_movies = n_movies
        self.embedding_dim = embedding_dim

        # Movie ID embedding
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        # Feature transformation
        self.feature_transform = nn.Linear(feature_dim, 32)

        # Dense layers
        input_dim = embedding_dim + 32
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        nn.init.xavier_uniform_(self.feature_transform.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        # Initialize biases to zero
        nn.init.zeros_(self.feature_transform.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self, movie_ids: torch.Tensor, movie_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through movie tower.

        Args:
            movie_ids: Movie IDs (batch_size,)
            movie_features: Movie features (batch_size, feature_dim)

        Returns:
            Movie embeddings (batch_size, embedding_dim)
        """
        # Get movie ID embedding
        movie_emb = self.movie_embedding(movie_ids)

        # Transform movie features
        if movie_features is not None:
            features = F.relu(self.feature_transform(movie_features))
            # Concatenate ID embedding and features
            x = torch.cat([movie_emb, features], dim=1)
        else:
            # Use only ID embedding (pad to expected dimension)
            padding = torch.zeros(
                movie_emb.size(0), 32, device=movie_emb.device, dtype=movie_emb.dtype
            )
            x = torch.cat([movie_emb, padding], dim=1)

        # Dense layers with batch norm and dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # L2 normalization for cosine similarity
        x = F.normalize(x, p=2, dim=1)

        return x


class TwoTowerModel(nn.Module):
    """
    Two-Tower Recommendation Model.

    Combines user and movie towers to predict ratings via similarity scoring.
    Supports both cosine similarity and dot product.
    """

    def __init__(
        self,
        n_users: int,
        n_movies: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        user_feature_dim: int = 30,
        movie_feature_dim: int = 12,
        dropout_rate: float = 0.2,
        similarity_method: str = "cosine",
    ):
        """
        Initialize Two-Tower Model.

        Args:
            n_users: Number of unique users
            n_movies: Number of unique movies
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden layer dimension
            user_feature_dim: Number of user features
            movie_feature_dim: Number of movie features
            dropout_rate: Dropout probability
            similarity_method: 'cosine' or 'dot_product'
        """
        super().__init__()

        self.n_users = n_users
        self.n_movies = n_movies
        self.embedding_dim = embedding_dim
        self.similarity_method = similarity_method

        # User tower
        self.user_tower = UserTower(
            n_users=n_users,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            feature_dim=user_feature_dim,
            dropout_rate=dropout_rate,
        )

        # Movie tower
        self.movie_tower = MovieTower(
            n_movies=n_movies,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            feature_dim=movie_feature_dim,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        movie_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        movie_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Two-Tower model.

        Args:
            user_ids: User IDs (batch_size,)
            movie_ids: Movie IDs (batch_size,)
            user_features: User features (batch_size, user_feature_dim)
            movie_features: Movie features (batch_size, movie_feature_dim)

        Returns:
            Tuple of:
                - predicted_ratings: Rating predictions (batch_size,)
                - user_embeddings: User embeddings (batch_size, embedding_dim)
                - movie_embeddings: Movie embeddings (batch_size, embedding_dim)
        """
        # Get embeddings from both towers
        user_emb = self.user_tower(user_ids, user_features)
        movie_emb = self.movie_tower(movie_ids, movie_features)

        # Compute similarity
        if self.similarity_method == "cosine":
            # Cosine similarity (embeddings already normalized)
            similarity = F.cosine_similarity(user_emb, movie_emb, dim=1)
        elif self.similarity_method == "dot_product":
            # Dot product
            similarity = (user_emb * movie_emb).sum(dim=1)
        else:
            raise ValueError(
                f"Unknown similarity method: {self.similarity_method}. "
                "Use 'cosine' or 'dot_product'"
            )

        # Scale similarity to rating range [0.5, 5.0]
        # Cosine similarity is in [-1, 1], map to [0.5, 5.0]
        predicted_ratings = 0.5 + 4.5 * (similarity + 1) / 2

        return predicted_ratings, user_emb, movie_emb

    def get_user_embedding(
        self, user_id: torch.Tensor, user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get embedding for a single user or batch of users.

        Args:
            user_id: User ID(s)
            user_features: User features

        Returns:
            User embedding(s)
        """
        return self.user_tower(user_id, user_features)

    def get_movie_embedding(
        self, movie_id: torch.Tensor, movie_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get embedding for a single movie or batch of movies.

        Args:
            movie_id: Movie ID(s)
            movie_features: Movie features

        Returns:
            Movie embedding(s)
        """
        return self.movie_tower(movie_id, movie_features)


def create_model(config: dict) -> TwoTowerModel:
    """
    Create Two-Tower model from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized TwoTowerModel
    """
    return TwoTowerModel(
        n_users=config["n_users"],
        n_movies=config["n_movies"],
        embedding_dim=config.get("embedding_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
        user_feature_dim=config.get("user_feature_dim", 30),
        movie_feature_dim=config.get("movie_feature_dim", 12),
        dropout_rate=config.get("dropout_rate", 0.2),
        similarity_method=config.get("similarity_method", "cosine"),
    )


if __name__ == "__main__":
    # Test model creation
    print("Testing Two-Tower Model...")

    config = {
        "n_users": 138000,
        "n_movies": 62000,
        "embedding_dim": 128,
        "hidden_dim": 256,
        "user_feature_dim": 30,
        "movie_feature_dim": 12,
    }

    model = create_model(config)

    # Test forward pass
    batch_size = 32
    user_ids = torch.randint(0, config["n_users"], (batch_size,))
    movie_ids = torch.randint(0, config["n_movies"], (batch_size,))

    ratings, user_emb, movie_emb = model(user_ids, movie_ids)

    print(f"✓ Model created successfully")
    print(f"  - User embeddings shape: {user_emb.shape}")
    print(f"  - Movie embeddings shape: {movie_emb.shape}")
    print(f"  - Predicted ratings shape: {ratings.shape}")
    print(f"  - Rating range: [{ratings.min():.2f}, {ratings.max():.2f}]")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
