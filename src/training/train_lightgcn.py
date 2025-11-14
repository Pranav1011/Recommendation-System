"""
Training Script for LightGCN Recommendation Model

Trains LightGCN with:
- Graph construction and normalization
- BPR loss optimization
- MLflow experiment tracking
- Early stopping and checkpointing
- Comprehensive ranking evaluation
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from src.data.graph_builder import build_graph
from src.models.lightgcn import create_lightgcn_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Simple metric functions for single user
def compute_recall(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Compute recall for single user."""
    if len(ground_truth) == 0:
        return 0.0
    hits = len(set(predictions) & set(ground_truth))
    return hits / len(ground_truth)


def compute_precision(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Compute precision for single user."""
    if len(predictions) == 0:
        return 0.0
    hits = len(set(predictions) & set(ground_truth))
    return hits / len(predictions)


def compute_ndcg(ground_truth: np.ndarray, predictions: np.ndarray, k: int) -> float:
    """Compute NDCG@K for single user."""
    # Create relevance scores (1 if in ground truth, 0 otherwise)
    relevance = np.array(
        [1.0 if item in ground_truth else 0.0 for item in predictions[:k]]
    )

    if relevance.sum() == 0:
        return 0.0

    # DCG
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

    # IDCG (ideal DCG)
    ideal_relevance = np.ones(min(len(ground_truth), k))
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def compute_hit_rate(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Compute hit rate (1 if any hit, 0 otherwise)."""
    return 1.0 if len(set(predictions) & set(ground_truth)) > 0 else 0.0


class LightGCNTrainer:
    """Trainer for LightGCN recommendation model."""

    def __init__(self, config: Dict):
        """
        Initialize LightGCN trainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("=" * 80)
        logger.info("LightGCN Trainer Initialization")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device}")

        # Set random seed
        torch.manual_seed(config.get("seed", 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.get("seed", 42))

        # Build graph
        self._build_graph()

        # Create model
        self._create_model()

        # Create optimizer and scheduler
        self._create_optimizer()

        # Load test data for evaluation
        self._load_test_data()

        # Training state
        self.best_metric = 0.0
        self.patience_counter = 0
        self.current_epoch = 0

        logger.info("Trainer initialized successfully!")
        logger.info("=" * 80)

    def _build_graph(self):
        """Build user-item bipartite graph."""
        logger.info("\n[1/4] Building Graph...")

        data_dir = Path(self.config["data_dir"])
        train_path = data_dir / "train_ratings.parquet"

        # Check cache
        graph_cache_dir = Path(self.config.get("graph_cache_dir", "data/graph_cache"))
        graph_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = graph_cache_dir / "lightgcn_graph.pt"

        if cache_file.exists():
            logger.info(f"Loading cached graph from {cache_file}")
            cache = torch.load(cache_file, weights_only=False)
            self.graph = cache["graph"]
            self.user_to_idx = cache["user_to_idx"]
            self.item_to_idx = cache["item_to_idx"]
            self.n_users = cache["n_users"]
            self.n_items = cache["n_items"]
            logger.info("Graph loaded from cache!")
        else:
            logger.info("Building graph from scratch...")
            graph_obj = build_graph(
                n_users=self.config["n_users"],
                n_items=self.config["n_movies"],
                train_ratings_path=str(train_path),
                min_rating=self.config["model"].get("rating_threshold", 4.0),
            )

            # Get sparse tensor
            indices, values, size = graph_obj.get_sparse_graph()
            self.graph = torch.sparse.FloatTensor(indices, values, size)

            # Store mappings
            self.user_to_idx = graph_obj.user_to_idx
            self.item_to_idx = graph_obj.item_to_idx
            self.n_users = graph_obj.n_users
            self.n_items = graph_obj.n_items

            # Cache the graph
            logger.info(f"Saving graph to {cache_file}")
            torch.save(
                {
                    "graph": self.graph,
                    "user_to_idx": self.user_to_idx,
                    "item_to_idx": self.item_to_idx,
                    "n_users": self.n_users,
                    "n_items": self.n_items,
                },
                cache_file,
            )
            logger.info("Graph cached!")

        # Move graph to device
        self.graph = self.graph.to(self.device)

    def _create_model(self):
        """Create LightGCN model."""
        logger.info("\n[2/4] Creating Model...")

        # Create model config
        model_config = {
            "n_users": self.n_users,
            "n_movies": self.n_items,
            "embedding_dim": self.config["model"]["embedding_dim"],
            "n_layers": self.config["model"]["n_layers"],
            "dropout_rate": self.config["model"].get("dropout_rate", 0.0),
        }

        self.model = create_lightgcn_model(model_config)
        self.model.to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model created with {n_params:,} parameters")

    def _create_optimizer(self):
        """Create optimizer and learning rate scheduler."""
        logger.info("\n[3/4] Creating Optimizer...")

        # Optimizer
        opt_config = self.config["optimizer"]
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=opt_config["learning_rate"],
            weight_decay=opt_config.get("weight_decay", 0.0),
        )

        # Learning rate scheduler
        sched_config = self.config.get("scheduler", {})
        if sched_config.get("type") == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config.get("mode", "max"),
                factor=sched_config.get("factor", 0.5),
                patience=sched_config.get("patience", 3),
                min_lr=sched_config.get("min_lr", 1e-6),
            )
        else:
            self.scheduler = None

        logger.info(f"Optimizer: Adam (LR={opt_config['learning_rate']})")
        if self.scheduler:
            logger.info("Scheduler: ReduceLROnPlateau")

    def _load_test_data(self):
        """Load test data for evaluation."""
        logger.info("\n[4/4] Loading Train & Test Data...")

        data_dir = Path(self.config["data_dir"])

        # Load training data for BPR sampling
        train_path = data_dir / "train_ratings.parquet"
        train_df = pd.read_parquet(train_path)

        min_rating = self.config["model"].get("rating_threshold", 4.0)
        train_df = train_df[train_df["rating"] >= min_rating].copy()

        # Map to indices
        train_df["user_idx"] = train_df["userId"].map(self.user_to_idx)
        train_df["item_idx"] = train_df["movieId"].map(self.item_to_idx)
        train_df = train_df.dropna(subset=["user_idx", "item_idx"])
        train_df["user_idx"] = train_df["user_idx"].astype(int)
        train_df["item_idx"] = train_df["item_idx"].astype(int)

        self.train_df = train_df

        # Group by user for efficient BPR sampling
        self.train_interactions = (
            train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )
        logger.info(
            f"Train data: {len(train_df):,} interactions, {len(self.train_interactions):,} users"
        )

        # Load test data
        test_path = data_dir / "test_ratings.parquet"
        test_df = pd.read_parquet(test_path)
        logger.info(f"Loaded {len(test_df):,} test ratings")

        # Filter high ratings only (same as training)
        test_df = test_df[test_df["rating"] >= min_rating].copy()
        logger.info(
            f"After filtering (rating >= {min_rating}): {len(test_df):,} interactions"
        )

        # Map to indices
        test_df["user_idx"] = test_df["userId"].map(self.user_to_idx)
        test_df["item_idx"] = test_df["movieId"].map(self.item_to_idx)

        # Remove unmapped entries
        test_df = test_df.dropna(subset=["user_idx", "item_idx"])
        test_df["user_idx"] = test_df["user_idx"].astype(int)
        test_df["item_idx"] = test_df["item_idx"].astype(int)

        self.test_df = test_df
        logger.info(f"Test data ready: {len(test_df):,} interactions")

        # Group by user for efficient evaluation
        self.test_interactions = (
            test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
        )
        logger.info(f"Test users: {len(self.test_interactions):,}")

    def _create_train_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create a training batch with BPR sampling.

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (users, pos_items, neg_items)
        """
        # Sample random users from those who have interactions
        user_indices = list(self.train_interactions.keys())
        sampled_users = np.random.choice(user_indices, size=batch_size, replace=True)

        # For each user, sample positive and negative items
        pos_items = []
        neg_items = []

        for user_idx in sampled_users:
            # Get user's positive items
            user_pos_items = self.train_interactions[user_idx]

            # Sample positive item
            pos_item = np.random.choice(list(user_pos_items))

            # Sample negative item (not in user's positives)
            neg_item = np.random.randint(0, self.n_items)
            while neg_item in user_pos_items:
                neg_item = np.random.randint(0, self.n_items)

            pos_items.append(pos_item)
            neg_items.append(neg_item)

        users = torch.tensor(sampled_users, device=self.device, dtype=torch.long)
        pos_items = torch.tensor(pos_items, device=self.device, dtype=torch.long)
        neg_items = torch.tensor(neg_items, device=self.device, dtype=torch.long)

        return users, pos_items, neg_items

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        batch_size = self.config["training"]["batch_size"]
        n_batches = len(self.train_df) // batch_size  # Approximate

        total_loss = 0.0
        total_bpr_loss = 0.0
        total_reg_loss = 0.0

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch}")
        for _ in pbar:
            # Create batch
            users, pos_items, neg_items = self._create_train_batch(batch_size)

            # Forward pass
            bpr_loss, reg_loss = self.model.bpr_loss(
                users,
                pos_items,
                neg_items,
                self.graph,
                reg_weight=self.config["loss"].get("reg_weight", 1e-4),
            )

            loss = bpr_loss + reg_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            max_grad_norm = self.config["training_loop"].get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_bpr_loss += bpr_loss.item()
            total_reg_loss += reg_loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "bpr": bpr_loss.item(),
                    "reg": reg_loss.item(),
                }
            )

        # Average metrics
        avg_loss = total_loss / n_batches
        avg_bpr_loss = total_bpr_loss / n_batches
        avg_reg_loss = total_reg_loss / n_batches

        return {
            "loss": avg_loss,
            "bpr_loss": avg_bpr_loss,
            "reg_loss": avg_reg_loss,
        }

    @torch.no_grad()
    def evaluate(self, k_values=[10, 20, 50]) -> Dict[str, float]:
        """
        Fast GPU-based evaluation on test set.

        Args:
            k_values: List of K values for metrics

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        # Get all user and item embeddings (on GPU)
        user_all_emb, item_all_emb = self.model(self.graph)

        metrics = {}

        # Sample 100 random users for balanced speed/accuracy
        eval_users = np.random.choice(
            list(self.test_interactions.keys()),
            size=min(100, len(self.test_interactions)),
            replace=False,
        )

        for k in k_values:
            recalls = []
            ndcgs = []

            for user_idx in eval_users:
                # Get user embedding (stays on GPU)
                user_emb = user_all_emb[user_idx]

                # Compute scores for all items ON GPU (fast!)
                scores = torch.matmul(item_all_emb, user_emb)

                # Get top-K items (on GPU, then transfer)
                _, top_k_indices = torch.topk(scores, k)
                top_k_items = top_k_indices.cpu().numpy()

                # Get ground truth
                gt_items = np.array(self.test_interactions[user_idx])

                # Compute metrics
                recalls.append(compute_recall(gt_items, top_k_items))
                ndcgs.append(compute_ndcg(gt_items, top_k_items, k))

            # Average metrics
            metrics[f"recall@{k}"] = np.mean(recalls)
            metrics[f"ndcg@{k}"] = np.mean(ndcgs)

        return metrics

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            path: Checkpoint path
            is_best: Whether this is the best model so far
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

        if is_best:
            best_path = path.parent / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")

    def train(self):
        """Main training loop."""
        logger.info("\n" + "=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)

        # MLflow setup
        mlflow_config = self.config.get("mlflow", {})
        mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "file:./mlruns"))
        mlflow.set_experiment(
            mlflow_config.get("experiment_name", "lightgcn_experiments")
        )

        with mlflow.start_run(run_name=mlflow_config.get("run_name", "lightgcn_run")):
            # Log config
            mlflow.log_params(
                {
                    "embedding_dim": self.config["model"]["embedding_dim"],
                    "n_layers": self.config["model"]["n_layers"],
                    "batch_size": self.config["training"]["batch_size"],
                    "learning_rate": self.config["optimizer"]["learning_rate"],
                    "reg_weight": self.config["loss"]["reg_weight"],
                }
            )

            # Training loop
            epochs = self.config["training_loop"]["epochs"]
            patience = self.config["training_loop"]["early_stopping_patience"]

            for epoch in range(1, epochs + 1):
                self.current_epoch = epoch

                # Train
                train_metrics = self.train_epoch(epoch)

                # Log training metrics
                for name, value in train_metrics.items():
                    mlflow.log_metric(f"train_{name}", value, step=epoch)

                # Evaluate
                if epoch % self.config["training_loop"].get("eval_every", 1) == 0:
                    eval_metrics = self.evaluate(
                        k_values=self.config["evaluation"]["k_values"]
                    )

                    # Log evaluation metrics (replace @ with _ for MLflow compatibility)
                    for name, value in eval_metrics.items():
                        mlflow_name = name.replace("@", "_at_")
                        mlflow.log_metric(mlflow_name, value, step=epoch)

                    # Print metrics
                    logger.info(f"\nEpoch {epoch}:")
                    logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
                    logger.info(f"  NDCG@10: {eval_metrics['ndcg@10']:.4f}")
                    logger.info(f"  Recall@10: {eval_metrics['recall@10']:.4f}")

                    # Check for improvement
                    current_metric = eval_metrics["ndcg@10"]
                    if current_metric > self.best_metric:
                        logger.info(f"  ✓ New best NDCG@10: {current_metric:.4f}")
                        self.best_metric = current_metric
                        self.patience_counter = 0

                        # Save best model
                        checkpoint_dir = Path(self.config["checkpoint_dir"])
                        self.save_checkpoint(
                            checkpoint_dir / f"lightgcn_epoch_{epoch}.pt",
                            is_best=True,
                        )
                    else:
                        self.patience_counter += 1
                        logger.info(
                            f"  No improvement ({self.patience_counter}/{patience})"
                        )

                    # Update scheduler
                    if self.scheduler:
                        self.scheduler.step(current_metric)
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        mlflow.log_metric("learning_rate", current_lr, step=epoch)

                    # Early stopping
                    if self.patience_counter >= patience:
                        logger.info(f"\nEarly stopping at epoch {epoch}")
                        break

            logger.info("\n" + "=" * 80)
            logger.info("Training Complete!")
            logger.info(f"Best NDCG@10: {self.best_metric:.4f}")
            logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train LightGCN model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config_lightgcn.json",
        help="Path to training config",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    logger.info(f"Loaded config from {args.config}")

    # Create trainer
    trainer = LightGCNTrainer(config)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
