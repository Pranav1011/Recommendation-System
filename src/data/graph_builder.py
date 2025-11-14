"""
Graph Builder for LightGCN

Constructs user-item bipartite graph from rating data.
Implements graph normalization for message passing.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

logger = logging.getLogger(__name__)


class UserItemGraph:
    """
    Build and manage user-item bipartite graph for LightGCN.

    The graph represents user-item interactions as edges.
    Implements symmetric normalization: D^(-1/2) * A * D^(-1/2)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        train_ratings_path: Path,
        min_rating: float = 4.0,
    ):
        """
        Initialize User-Item Graph.

        Args:
            n_users: Total number of users
            n_items: Total number of items
            train_ratings_path: Path to training ratings
            min_rating: Minimum rating to consider as interaction
        """
        self.n_users = n_users
        self.n_items = n_items
        self.min_rating = min_rating

        logger.info(f"Building user-item graph: {n_users} users, {n_items} items")
        logger.info(f"Min rating threshold: {min_rating}")

        # Load training data
        self._load_data(train_ratings_path)

        # Build adjacency matrix
        self._build_adjacency_matrix()

        # Compute normalized adjacency
        self._normalize_adjacency()

        logger.info("User-item graph ready!")

    def _load_data(self, ratings_path: Path) -> None:
        """Load and filter training ratings."""
        logger.info(f"Loading ratings from {ratings_path}")

        if not ratings_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

        # Load ratings
        self.ratings_df = pd.read_parquet(ratings_path)
        logger.info(f"Loaded {len(self.ratings_df):,} ratings")

        # Filter by minimum rating (implicit feedback)
        self.ratings_df = self.ratings_df[
            self.ratings_df["rating"] >= self.min_rating
        ].copy()
        logger.info(
            f"After filtering (rating >= {self.min_rating}): {len(self.ratings_df):,} interactions"
        )

        # Get unique user/item IDs
        self.user_ids = sorted(self.ratings_df["userId"].unique())
        self.item_ids = sorted(self.ratings_df["movieId"].unique())

        logger.info(f"Unique users: {len(self.user_ids):,}")
        logger.info(f"Unique items: {len(self.item_ids):,}")

        # Create ID to index mappings
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}

    def _build_adjacency_matrix(self) -> None:
        """
        Build adjacency matrix for user-item bipartite graph.

        Graph structure:
            Users (0 to n_users-1)
            Items (n_users to n_users+n_items-1)

        Adjacency matrix A is (n_users + n_items) x (n_users + n_items)
        """
        logger.info("Building adjacency matrix...")

        # Map user/item IDs to indices
        user_indices = self.ratings_df["userId"].map(self.user_to_idx).values
        item_indices = self.ratings_df["movieId"].map(self.item_to_idx).values

        # Create sparse matrix for user-item interactions
        # Shape: (n_users, n_items)
        n_interactions = len(user_indices)
        values = np.ones(n_interactions)

        user_item_matrix = sp.coo_matrix(
            (values, (user_indices, item_indices)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

        logger.info(
            f"User-item matrix: {user_item_matrix.shape}, "
            f"{user_item_matrix.nnz:,} edges"
        )

        # Build bipartite adjacency matrix
        # A = [[0,         R      ],
        #      [R^T,       0      ]]
        # where R is user-item interaction matrix

        # Top-right: user -> item edges
        top_right = user_item_matrix

        # Bottom-left: item -> user edges (transpose)
        bottom_left = user_item_matrix.T

        # Combine into full adjacency matrix
        # Using block matrix construction
        zero_users = sp.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
        zero_items = sp.csr_matrix((self.n_items, self.n_items), dtype=np.float32)

        # Stack blocks
        top = sp.hstack([zero_users, top_right])
        bottom = sp.hstack([bottom_left, zero_items])
        self.adjacency = sp.vstack([top, bottom]).tocsr()

        logger.info(f"Adjacency matrix shape: {self.adjacency.shape}")
        logger.info(f"Total edges (both directions): {self.adjacency.nnz:,}")

        # Sparsity
        total_possible = self.adjacency.shape[0] * self.adjacency.shape[1]
        sparsity = 1 - (self.adjacency.nnz / total_possible)
        logger.info(f"Graph sparsity: {sparsity:.4%}")

    def _normalize_adjacency(self) -> None:
        """
        Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)

        This is the symmetric normalization used in LightGCN.
        Ensures stable message passing.
        """
        logger.info("Normalizing adjacency matrix...")

        # Add self-loops (A + I)
        # This helps with gradient flow
        adj_with_self_loops = self.adjacency + sp.eye(self.adjacency.shape[0])

        # Compute degree matrix D
        rowsum = np.array(adj_with_self_loops.sum(axis=1)).flatten()

        # D^(-1/2)
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0  # Handle isolated nodes
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)

        # Normalize: D^(-1/2) * A * D^(-1/2)
        norm_adj = d_inv_sqrt_mat @ adj_with_self_loops @ d_inv_sqrt_mat

        self.norm_adj = norm_adj.tocoo()
        logger.info("Adjacency normalized!")

    def get_sparse_graph(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
        """
        Get normalized adjacency as PyTorch sparse tensor.

        Returns:
            Tuple of (indices, values, size) for sparse tensor
        """
        # Convert to COO format
        coo = self.norm_adj.tocoo()

        # Create indices tensor
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))

        # Create values tensor
        values = torch.FloatTensor(coo.data)

        # Shape
        size = torch.Size(coo.shape)

        return indices, values, size

    def get_dense_graph(self) -> torch.Tensor:
        """
        Get normalized adjacency as dense PyTorch tensor.

        Warning: Only use for small graphs (memory intensive).
        """
        return torch.FloatTensor(self.norm_adj.toarray())

    def get_user_interactions(self, user_idx: int) -> np.ndarray:
        """
        Get items that user has interacted with.

        Args:
            user_idx: User index (not user ID)

        Returns:
            Array of item indices
        """
        user_id = self.user_ids[user_idx]
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        item_indices = user_ratings["movieId"].map(self.item_to_idx).values
        return item_indices

    def get_item_interactions(self, item_idx: int) -> np.ndarray:
        """
        Get users who interacted with item.

        Args:
            item_idx: Item index (not item ID)

        Returns:
            Array of user indices
        """
        item_id = self.item_ids[item_idx]
        item_ratings = self.ratings_df[self.ratings_df["movieId"] == item_id]
        user_indices = item_ratings["userId"].map(self.user_to_idx).values
        return user_indices

    def get_statistics(self) -> Dict[str, float]:
        """Get graph statistics."""
        # User degree stats
        user_degrees = np.diff(self.adjacency[: self.n_users].tocsr().indptr)

        # Item degree stats
        item_degrees = np.diff(self.adjacency[self.n_users :].tocsr().indptr)

        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_interactions": len(self.ratings_df),
            "avg_user_degree": float(np.mean(user_degrees)),
            "avg_item_degree": float(np.mean(item_degrees)),
            "max_user_degree": int(np.max(user_degrees)),
            "max_item_degree": int(np.max(item_degrees)),
            "sparsity": 1 - (2 * len(self.ratings_df) / (self.n_users * self.n_items)),
        }


def build_graph(
    n_users: int,
    n_items: int,
    train_ratings_path: str,
    min_rating: float = 4.0,
) -> UserItemGraph:
    """
    Build user-item graph from ratings.

    Args:
        n_users: Number of users
        n_items: Number of items
        train_ratings_path: Path to training ratings
        min_rating: Minimum rating for interaction

    Returns:
        UserItemGraph instance
    """
    graph = UserItemGraph(
        n_users=n_users,
        n_items=n_items,
        train_ratings_path=Path(train_ratings_path),
        min_rating=min_rating,
    )

    # Log statistics
    stats = graph.get_statistics()
    logger.info("\nGraph Statistics:")
    logger.info(f"  Users: {stats['n_users']:,}")
    logger.info(f"  Items: {stats['n_items']:,}")
    logger.info(f"  Interactions: {stats['n_interactions']:,}")
    logger.info(f"  Avg user degree: {stats['avg_user_degree']:.1f}")
    logger.info(f"  Avg item degree: {stats['avg_item_degree']:.1f}")
    logger.info(f"  Sparsity: {stats['sparsity']:.4%}")

    return graph


if __name__ == "__main__":
    # Test graph building
    logging.basicConfig(level=logging.INFO)

    graph = build_graph(
        n_users=162541,
        n_items=59047,
        train_ratings_path="data/processed/train_ratings.parquet",
        min_rating=4.0,
    )

    # Get sparse representation
    indices, values, size = graph.get_sparse_graph()
    print(f"\nSparse graph: {size}")
    print(f"Indices shape: {indices.shape}")
    print(f"Values shape: {values.shape}")

    # Test user interactions
    user_items = graph.get_user_interactions(0)
    print(f"\nUser 0 interacted with {len(user_items)} items")

    print("\nGraph built successfully!")
