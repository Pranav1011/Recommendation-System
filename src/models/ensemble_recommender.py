"""
Ensemble Recommender System

Combines multiple models with weighted averaging for improved recommendation quality.
Supports LightGCN and Two-Tower models with configurable weights.

Architecture:
- LightGCN: Graph-based collaborative filtering (default weight: 0.7)
- Two-Tower: Deep learning with features (default weight: 0.3)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.graph_builder import build_graph
from src.models.lightgcn import create_lightgcn_model
from src.models.two_tower import create_model as create_two_tower_model

logger = logging.getLogger(__name__)


def load_twotower_user_mapping(train_ratings_path: Path) -> Dict[int, int]:
    """
    Load Two-Tower user ID to index mapping.

    Reconstructs the exact mapping used during Two-Tower training.
    The dataset creates sequential indices from sorted unique user IDs.

    Args:
        train_ratings_path: Path to train_ratings.parquet

    Returns:
        Dictionary mapping {user_id: sequential_index}
    """
    logger.info(f"Loading Two-Tower user mapping from {train_ratings_path}")
    ratings_df = pd.read_parquet(train_ratings_path)

    # This matches exactly what the dataset does
    unique_users = sorted(ratings_df["userId"].unique())
    user_to_idx = {int(uid): idx for idx, uid in enumerate(unique_users)}

    logger.info(f"Loaded Two-Tower user mapping: {len(user_to_idx)} users")
    logger.info(f"User ID range: {min(user_to_idx.keys())} to {max(user_to_idx.keys())}")
    logger.info(f"Index range: 0 to {len(user_to_idx) - 1}")

    return user_to_idx


def load_twotower_item_mapping(train_ratings_path: Path) -> Dict[int, int]:
    """
    Load Two-Tower item ID to index mapping.

    Reconstructs the exact mapping used during Two-Tower training.

    Args:
        train_ratings_path: Path to train_ratings.parquet

    Returns:
        Dictionary mapping {item_id: sequential_index}
    """
    logger.info(f"Loading Two-Tower item mapping from {train_ratings_path}")
    ratings_df = pd.read_parquet(train_ratings_path)

    unique_items = sorted(ratings_df["movieId"].unique())
    item_to_idx = {int(iid): idx for idx, iid in enumerate(unique_items)}

    logger.info(f"Loaded Two-Tower item mapping: {len(item_to_idx)} items")
    logger.info(f"Item ID range: {min(item_to_idx.keys())} to {max(item_to_idx.keys())}")
    logger.info(f"Index range: 0 to {len(item_to_idx) - 1}")

    return item_to_idx


class EnsembleRecommender:
    """
    Ensemble Recommender combining multiple models.

    Supports weighted combination of:
    - LightGCN (graph-based collaborative filtering)
    - Two-Tower (neural collaborative filtering with features)

    The ensemble creates a weighted average of embeddings from both models,
    combining the strengths of graph structure and deep feature learning.
    """

    def __init__(
        self,
        lightgcn_checkpoint: str,
        lightgcn_config: Union[str, Dict],
        twotower_checkpoint: str,
        twotower_config: Union[str, Dict],
        data_dir: str = "data/processed",
        features_dir: str = "data/features",
        lightgcn_weight: float = 0.7,
        twotower_weight: float = 0.3,
        device: str = "cuda",
    ):
        """
        Initialize Ensemble Recommender.

        Args:
            lightgcn_checkpoint: Path to LightGCN checkpoint
            lightgcn_config: Path to LightGCN config or config dict
            twotower_checkpoint: Path to Two-Tower checkpoint
            twotower_config: Path to Two-Tower config or config dict
            data_dir: Directory containing processed data
            features_dir: Directory containing feature data
            lightgcn_weight: Weight for LightGCN (default: 0.7)
            twotower_weight: Weight for Two-Tower (default: 0.3)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Ensemble Recommender on {self.device}")

        # Validate weights
        if not np.isclose(lightgcn_weight + twotower_weight, 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got {lightgcn_weight + twotower_weight}"
            )

        self.lightgcn_weight = lightgcn_weight
        self.twotower_weight = twotower_weight
        logger.info(
            f"Weights: LightGCN={lightgcn_weight:.2f}, TwoTower={twotower_weight:.2f}"
        )

        # Load configs
        self.lightgcn_config = self._load_config(lightgcn_config)
        self.twotower_config = self._load_config(twotower_config)

        # Store paths
        self.data_dir = Path(data_dir)
        self.features_dir = Path(features_dir)

        # Validate embedding dimensions
        self._validate_embedding_dims()

        # Load models
        self._load_lightgcn(lightgcn_checkpoint)
        self._load_twotower(twotower_checkpoint)

        # Load features (optional, for Two-Tower)
        self._load_features()

        # Generate embeddings
        self._generate_embeddings()

        logger.info("Ensemble Recommender initialized successfully!")

    def _load_config(self, config: Union[str, Dict]) -> Dict:
        """Load configuration from file or dict."""
        if isinstance(config, dict):
            return config

        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            return json.load(f)

    def _validate_embedding_dims(self) -> None:
        """Validate that embedding dimensions match between models."""
        lightgcn_dim = self.lightgcn_config["model"]["embedding_dim"]
        twotower_dim = self.twotower_config["embedding_dim"]

        if lightgcn_dim != twotower_dim:
            logger.warning(
                f"Embedding dimension mismatch: LightGCN={lightgcn_dim}, "
                f"TwoTower={twotower_dim}. Will use projection."
            )
            self.use_projection = True
            self.target_dim = max(lightgcn_dim, twotower_dim)
        else:
            self.use_projection = False
            self.target_dim = lightgcn_dim

    def _load_lightgcn(self, checkpoint_path: str) -> None:
        """Load LightGCN model and graph."""
        logger.info(f"Loading LightGCN from {checkpoint_path}")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"LightGCN checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Create model
        model_config = {
            "n_users": self.lightgcn_config["n_users"],
            "n_items": self.lightgcn_config["n_movies"],
            "embedding_dim": self.lightgcn_config["model"]["embedding_dim"],
            "n_layers": self.lightgcn_config["model"]["n_layers"],
            "dropout_rate": self.lightgcn_config["model"].get("dropout_rate", 0.0),
        }
        self.lightgcn_model = create_lightgcn_model(model_config)
        self.lightgcn_model.load_state_dict(checkpoint["model_state_dict"])
        self.lightgcn_model.to(self.device)
        self.lightgcn_model.eval()

        logger.info(f"LightGCN loaded (epoch {checkpoint['epoch']})")

        # Build or load graph
        self._load_graph()

        # Create projection if needed
        if (
            self.use_projection
            and self.lightgcn_config["model"]["embedding_dim"] < self.target_dim
        ):
            self.lightgcn_projection = torch.nn.Linear(
                self.lightgcn_config["model"]["embedding_dim"], self.target_dim
            ).to(self.device)
            torch.nn.init.xavier_uniform_(self.lightgcn_projection.weight)

    def _load_graph(self) -> None:
        """Load or build user-item graph for LightGCN."""
        # Check for cached graph
        graph_cache_dir = Path(
            self.lightgcn_config.get("graph_cache_dir", "data/graph_cache")
        )
        graph_cache_file = graph_cache_dir / "lightgcn_graph.pt"

        if graph_cache_file.exists():
            logger.info("Loading cached graph...")
            cache = torch.load(graph_cache_file, weights_only=False)
            self.graph = cache["graph"].to(self.device)
            self.user_to_idx = cache["user_to_idx"]
            self.item_to_idx = cache["item_to_idx"]
        else:
            logger.info("Building graph from scratch...")
            graph_obj = build_graph(
                n_users=self.lightgcn_config["n_users"],
                n_items=self.lightgcn_config["n_movies"],
                train_ratings_path=str(self.data_dir / "train_ratings.parquet"),
                min_rating=self.lightgcn_config["model"].get("rating_threshold", 4.0),
            )
            indices, values, size = graph_obj.get_sparse_graph()
            self.graph = torch.sparse.FloatTensor(indices, values, size).to(self.device)
            self.user_to_idx = graph_obj.user_to_idx
            self.item_to_idx = graph_obj.item_to_idx

            # Save cache
            graph_cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "graph": self.graph.cpu(),
                    "user_to_idx": self.user_to_idx,
                    "item_to_idx": self.item_to_idx,
                },
                graph_cache_file,
            )
            logger.info(f"Graph cached to {graph_cache_file}")

    def _load_twotower(self, checkpoint_path: str) -> None:
        """Load Two-Tower model."""
        logger.info(f"Loading Two-Tower from {checkpoint_path}")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Two-Tower checkpoint not found: {checkpoint_path}"
            )

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Detect actual feature dimensions from checkpoint
        state_dict = checkpoint["model_state_dict"]
        actual_user_feat_dim = state_dict.get("user_tower.feature_transform.weight", torch.zeros(32, 0)).shape[1]
        actual_movie_feat_dim = state_dict.get("movie_tower.feature_transform.weight", torch.zeros(32, 0)).shape[1]

        logger.info(f"Detected feature dims from checkpoint: user={actual_user_feat_dim}, movie={actual_movie_feat_dim}")

        # Override config with actual dimensions (update self.twotower_config)
        self.twotower_config["user_feature_dim"] = actual_user_feat_dim
        self.twotower_config["movie_feature_dim"] = actual_movie_feat_dim

        # Create model with corrected config
        self.twotower_model = create_two_tower_model(self.twotower_config)
        self.twotower_model.load_state_dict(checkpoint["model_state_dict"])
        self.twotower_model.to(self.device)
        self.twotower_model.eval()

        logger.info(f"Two-Tower loaded (epoch {checkpoint['epoch']})")

        # Load Two-Tower ID mappings
        train_ratings_path = self.data_dir / "train_ratings.parquet"
        self.twotower_user_to_idx = load_twotower_user_mapping(train_ratings_path)
        self.twotower_item_to_idx = load_twotower_item_mapping(train_ratings_path)

        # Create projection if needed
        if (
            self.use_projection
            and self.twotower_config["embedding_dim"] < self.target_dim
        ):
            self.twotower_projection = torch.nn.Linear(
                self.twotower_config["embedding_dim"], self.target_dim
            ).to(self.device)
            torch.nn.init.xavier_uniform_(self.twotower_projection.weight)

    def _load_features(self) -> None:
        """Load user and movie features for Two-Tower model."""
        # Check if model was trained with features
        user_feat_dim = self.twotower_config.get("user_feature_dim", 0)
        movie_feat_dim = self.twotower_config.get("movie_feature_dim", 0)

        if user_feat_dim == 0 and movie_feat_dim == 0:
            logger.info("Two-Tower model was trained without features. Skipping feature loading.")
            self.user_features = None
            self.movie_features = None
            return

        logger.info("Loading features...")

        # Load user features (only if needed)
        if user_feat_dim > 0:
            user_features_path = self.features_dir / "user_features.parquet"
            if user_features_path.exists():
                user_features_df = pd.read_parquet(user_features_path)
                # Align with user_to_idx mapping
                self.user_features = self._align_features(
                    user_features_df, self.user_to_idx, "userId"
                )
            else:
                logger.warning("User features not found, using None")
                self.user_features = None
        else:
            self.user_features = None

        # Load movie features (only if needed)
        if movie_feat_dim > 0:
            movie_features_path = self.features_dir / "movie_features.parquet"
            if movie_features_path.exists():
                movie_features_df = pd.read_parquet(movie_features_path)
                # Align with item_to_idx mapping
                self.movie_features = self._align_features(
                    movie_features_df, self.item_to_idx, "movieId"
                )
            else:
                logger.warning("Movie features not found, using None")
                self.movie_features = None
        else:
            self.movie_features = None

    def _align_features(
        self, features_df: pd.DataFrame, id_to_idx: Dict[int, int], id_col: str
    ) -> torch.Tensor:
        """Align features with index mapping."""
        # Create feature matrix aligned with indices
        n_entities = len(id_to_idx)
        feature_cols = [col for col in features_df.columns if col != id_col]
        n_features = len(feature_cols)

        feature_matrix = np.zeros((n_entities, n_features), dtype=np.float32)

        for entity_id, idx in id_to_idx.items():
            if entity_id in features_df[id_col].values:
                row = features_df[features_df[id_col] == entity_id][
                    feature_cols
                ].values[0]
                feature_matrix[idx] = row

        return torch.tensor(feature_matrix, dtype=torch.float32, device=self.device)

    def _generate_embeddings(self) -> None:
        """Generate embeddings from both models."""
        logger.info("Generating embeddings from models...")

        with torch.no_grad():
            # LightGCN embeddings
            logger.info("Computing LightGCN embeddings...")
            lightgcn_user_emb, lightgcn_item_emb = self.lightgcn_model(self.graph)

            # Project if needed
            if self.use_projection and hasattr(self, "lightgcn_projection"):
                lightgcn_user_emb = self.lightgcn_projection(lightgcn_user_emb)
                lightgcn_item_emb = self.lightgcn_projection(lightgcn_item_emb)

            # Two-Tower embeddings
            logger.info("Computing Two-Tower embeddings...")

            # CRITICAL FIX: Match LightGCN's full embedding size
            # LightGCN returns embeddings for all n_users (162,541), not just those in graph
            n_users_full = lightgcn_user_emb.shape[0]  # Full LightGCN size
            n_items_full = lightgcn_item_emb.shape[0]  # Full LightGCN size

            # Compute user embeddings - ALIGNED with LightGCN shape
            twotower_user_emb = torch.zeros(
                n_users_full, self.twotower_config["embedding_dim"], device=self.device
            )
            batch_size = 1024

            # Create list of actual user IDs from LightGCN graph
            lightgcn_user_ids = sorted(self.user_to_idx.keys())

            n_users = len(self.user_to_idx)  # Users in LightGCN graph
            for i in range(0, n_users, batch_size):
                end_i = min(i + batch_size, n_users)

                # Get actual user IDs from LightGCN mapping
                batch_user_ids = lightgcn_user_ids[i:end_i]

                # CRITICAL FIX: Map to Two-Tower indices
                twotower_indices = []
                for uid in batch_user_ids:
                    if uid in self.twotower_user_to_idx:
                        twotower_indices.append(self.twotower_user_to_idx[uid])
                    else:
                        # User not in Two-Tower training - skip or use default
                        logger.warning(f"User {uid} not in Two-Tower mapping, using index 0")
                        twotower_indices.append(0)

                user_ids_tensor = torch.tensor(twotower_indices, device=self.device)

                # Get features if available
                user_feats = None
                if self.user_features is not None:
                    user_feats = self.user_features[i:end_i]

                # Get Two-Tower embeddings using INDICES
                twotower_user_emb[i:end_i] = self.twotower_model.get_user_embedding(
                    user_ids_tensor, user_feats
                )

            # Compute item embeddings - ALIGNED with LightGCN shape
            twotower_item_emb = torch.zeros(
                n_items_full, self.twotower_config["embedding_dim"], device=self.device
            )

            # Create list of actual item IDs from LightGCN graph
            lightgcn_item_ids = sorted(self.item_to_idx.keys())

            n_items = len(self.item_to_idx)  # Items in LightGCN graph
            for i in range(0, n_items, batch_size):
                end_i = min(i + batch_size, n_items)

                # Get actual item IDs from LightGCN mapping
                batch_item_ids = lightgcn_item_ids[i:end_i]

                # CRITICAL FIX: Map to Two-Tower indices
                twotower_indices = []
                for iid in batch_item_ids:
                    if iid in self.twotower_item_to_idx:
                        twotower_indices.append(self.twotower_item_to_idx[iid])
                    else:
                        logger.warning(f"Item {iid} not in Two-Tower mapping, using index 0")
                        twotower_indices.append(0)

                item_ids_tensor = torch.tensor(twotower_indices, device=self.device)

                # Get features if available
                item_feats = None
                if self.movie_features is not None:
                    item_feats = self.movie_features[i:end_i]

                # Get Two-Tower embeddings using INDICES
                twotower_item_emb[i:end_i] = self.twotower_model.get_movie_embedding(
                    item_ids_tensor, item_feats
                )

            # Project if needed
            if self.use_projection and hasattr(self, "twotower_projection"):
                twotower_user_emb = self.twotower_projection(twotower_user_emb)
                twotower_item_emb = self.twotower_projection(twotower_item_emb)

            # Weighted combination - NOW ALIGNED!
            logger.info("Combining embeddings...")
            logger.info(f"LightGCN shapes: users={lightgcn_user_emb.shape}, items={lightgcn_item_emb.shape}")
            logger.info(f"Two-Tower shapes: users={twotower_user_emb.shape}, items={twotower_item_emb.shape}")

            self.user_embeddings = (
                self.lightgcn_weight * lightgcn_user_emb
                + self.twotower_weight * twotower_user_emb
            )
            self.item_embeddings = (
                self.lightgcn_weight * lightgcn_item_emb
                + self.twotower_weight * twotower_item_emb
            )

            # L2 normalize for cosine similarity
            self.user_embeddings = F.normalize(self.user_embeddings, p=2, dim=1)
            self.item_embeddings = F.normalize(self.item_embeddings, p=2, dim=1)

        logger.info(
            f"Embeddings generated: users={self.user_embeddings.shape}, items={self.item_embeddings.shape}"
        )

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """
        Get ensemble embedding for a user.

        Args:
            user_id: User ID (original ID, not index)

        Returns:
            User embedding vector

        Raises:
            KeyError: If user_id not found
        """
        if user_id not in self.user_to_idx:
            raise KeyError(f"User ID {user_id} not found in training data")

        user_idx = self.user_to_idx[user_id]
        return self.user_embeddings[user_idx].cpu().numpy()

    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """
        Get ensemble embedding for an item.

        Args:
            item_id: Item ID (original ID, not index)

        Returns:
            Item embedding vector

        Raises:
            KeyError: If item_id not found
        """
        if item_id not in self.item_to_idx:
            raise KeyError(f"Item ID {item_id} not found in training data")

        item_idx = self.item_to_idx[item_id]
        return self.item_embeddings[item_idx].cpu().numpy()

    def predict_scores(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """
        Predict scores for user-item pairs.

        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs (must match length of user_ids)

        Returns:
            Array of predicted scores

        Raises:
            ValueError: If lengths don't match
            KeyError: If any ID not found
        """
        if len(user_ids) != len(item_ids):
            raise ValueError("user_ids and item_ids must have same length")

        # Convert to indices
        user_indices = [self.user_to_idx[uid] for uid in user_ids]
        item_indices = [self.item_to_idx[iid] for iid in item_ids]

        # Get embeddings
        user_emb = self.user_embeddings[user_indices]
        item_emb = self.item_embeddings[item_indices]

        # Compute scores (cosine similarity)
        scores = (user_emb * item_emb).sum(dim=1)

        return scores.cpu().numpy()

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        Get top-K recommendations for a user.

        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_seen: Whether to exclude items user has seen in training

        Returns:
            Tuple of (item_ids, scores) - both lists of length k

        Raises:
            KeyError: If user_id not found
        """
        if user_id not in self.user_to_idx:
            raise KeyError(f"User ID {user_id} not found in training data")

        user_idx = self.user_to_idx[user_id]
        user_emb = self.user_embeddings[user_idx]

        # Compute scores for all items
        scores = torch.matmul(self.item_embeddings, user_emb)

        # Exclude seen items
        if exclude_seen:
            # Get items user has interacted with
            train_ratings = pd.read_parquet(self.data_dir / "train_ratings.parquet")
            user_ratings = train_ratings[train_ratings["userId"] == user_id]
            seen_items = user_ratings["movieId"].values

            # Mask seen items
            for item_id in seen_items:
                if item_id in self.item_to_idx:
                    item_idx = self.item_to_idx[item_id]
                    scores[item_idx] = -float("inf")

        # Get top-K
        top_k_scores, top_k_indices = torch.topk(scores, k)

        # Convert to original item IDs
        idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}
        recommended_items = [idx_to_item[idx.item()] for idx in top_k_indices]
        recommended_scores = top_k_scores.cpu().numpy().tolist()

        return recommended_items, recommended_scores

    def get_similar_items(
        self, item_id: int, k: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Get top-K similar items.

        Args:
            item_id: Item ID
            k: Number of similar items to return

        Returns:
            Tuple of (item_ids, similarity_scores)

        Raises:
            KeyError: If item_id not found
        """
        if item_id not in self.item_to_idx:
            raise KeyError(f"Item ID {item_id} not found")

        item_idx = self.item_to_idx[item_id]
        item_emb = self.item_embeddings[item_idx]

        # Compute similarity with all items
        similarities = torch.matmul(self.item_embeddings, item_emb)

        # Exclude the item itself
        similarities[item_idx] = -float("inf")

        # Get top-K
        top_k_scores, top_k_indices = torch.topk(similarities, k)

        # Convert to original item IDs
        idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}
        similar_items = [idx_to_item[idx.item()] for idx in top_k_indices]
        similarity_scores = top_k_scores.cpu().numpy().tolist()

        return similar_items, similarity_scores

    def save_embeddings(self, output_dir: str) -> None:
        """
        Save ensemble embeddings to disk.

        Args:
            output_dir: Directory to save embeddings
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving embeddings to {output_dir}")

        # Save as numpy arrays
        np.save(output_dir / "user_embeddings.npy", self.user_embeddings.cpu().numpy())
        np.save(output_dir / "item_embeddings.npy", self.item_embeddings.cpu().numpy())

        # Save mappings (convert numpy int types to Python int for JSON)
        with open(output_dir / "user_to_idx.json", "w") as f:
            user_to_idx_serializable = {int(k): int(v) for k, v in self.user_to_idx.items()}
            json.dump(user_to_idx_serializable, f)
        with open(output_dir / "item_to_idx.json", "w") as f:
            item_to_idx_serializable = {int(k): int(v) for k, v in self.item_to_idx.items()}
            json.dump(item_to_idx_serializable, f)

        logger.info("Embeddings saved successfully!")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize ensemble
    ensemble = EnsembleRecommender(
        lightgcn_checkpoint="models/checkpoints/best_model_lightgcn_optimized.pt",
        lightgcn_config="configs/train_config_lightgcn_optimized.json",
        twotower_checkpoint="models/checkpoints/best_model_optimized.pt",
        twotower_config="configs/train_config_bpr_optimized.json",
        lightgcn_weight=0.7,
        twotower_weight=0.3,
    )

    # Get recommendations for user
    user_id = 1
    items, scores = ensemble.recommend(user_id, k=10)
    print(f"\nTop 10 recommendations for user {user_id}:")
    for item, score in zip(items, scores):
        print(f"  Item {item}: {score:.4f}")

    # Get similar items
    item_id = items[0]
    similar_items, sim_scores = ensemble.get_similar_items(item_id, k=5)
    print(f"\nTop 5 similar items to {item_id}:")
    for item, score in zip(similar_items, sim_scores):
        print(f"  Item {item}: {score:.4f}")

    # Save embeddings
    ensemble.save_embeddings("data/embeddings_ensemble")
    print("\nEmbeddings saved!")
