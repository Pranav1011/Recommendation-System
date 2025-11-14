"""
Training Script for Two-Tower Recommendation Model

Trains the model with:
- MLflow experiment tracking
- Early stopping
- Model checkpointing
- Comprehensive evaluation
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import mlflow
import mlflow.pytorch
import torch
import torch.optim as optim
from tqdm import tqdm

from src.models.losses import create_loss_function
from src.models.two_tower import create_model
from src.training.dataset import create_dataloaders
from src.training.hard_negative_sampler import HardNegativeSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for Two-Tower recommendation model."""

    def __init__(self, config: Dict):
        """
        Initialize trainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info(f"Using device: {self.device}")

        # Load data
        self._load_data()

        # Create model
        self._create_model()

        # Create loss function and optimizer
        self._create_loss_and_optimizer()

        # Training state
        self.best_val_loss = float("in")
        self.patience_counter = 0
        self.current_epoch = 0

    def _load_data(self):
        """Load datasets and create dataloaders."""
        logger.info("Loading datasets...")

        data_dir = Path(self.config["data_dir"])
        features_dir = Path(self.config["features_dir"])

        train_path = data_dir / "train_ratings.parquet"
        test_path = data_dir / "test_ratings.parquet"
        movies_path = data_dir / "movies.parquet"
        user_features_path = features_dir / "user_features.parquet"
        movie_features_path = features_dir / "movie_features.parquet"

        # Initialize hard negative sampler if requested
        hard_negative_sampler = None
        if self.config.get("hard_negative_sampling", False):
            logger.info("Initializing Hard Negative Sampler...")
            hard_negative_sampler = HardNegativeSampler(
                train_ratings_path=train_path,
                movies_path=movies_path,
                n_movies=self.config["n_movies"],
                popularity_weight=self.config.get("popularity_weight", 0.7),
                genre_weight=self.config.get("genre_weight", 0.2),
                random_weight=self.config.get("random_weight", 0.1),
            )

        self.train_loader, self.test_loader = create_dataloaders(
            train_ratings_path=train_path,
            test_ratings_path=test_path,
            user_features_path=user_features_path,
            movie_features_path=movie_features_path,
            batch_size=self.config.get("batch_size", 512),
            num_workers=self.config.get("num_workers", 4),
            negative_sampling=self.config.get("negative_sampling", False),
            n_negatives=self.config.get("n_negatives", 4),
            hard_negative_sampler=hard_negative_sampler,
        )

        # Get dataset info
        dataset = self.train_loader.dataset
        self.n_users = dataset.get_num_users()
        self.n_movies = dataset.get_num_movies()
        user_feat_dim, movie_feat_dim = dataset.get_feature_dims()

        logger.info("Dataset info:")
        logger.info(f"  - Users: {self.n_users:,}")
        logger.info(f"  - Movies: {self.n_movies:,}")
        logger.info(f"  - User feature dim: {user_feat_dim}")
        logger.info(f"  - Movie feature dim: {movie_feat_dim}")

        # Update config with feature dims only - TRUST config for n_users/n_movies!
        # self.config["n_users"] = self.n_users  # DISABLED - use config value
        # self.config["n_movies"] = self.n_movies  # DISABLED - use config value
        self.config["user_feature_dim"] = user_feat_dim
        self.config["movie_feature_dim"] = movie_feat_dim

    def _create_model(self):
        """Create Two-Tower model."""
        logger.info("Creating model...")

        self.model = create_model(self.config)
        self.model.to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model created with {n_params:,} parameters")

    def _create_loss_and_optimizer(self):
        """Create loss function and optimizer."""
        # Loss function
        self.criterion = create_loss_function(self.config)

        # Optimizer
        optimizer_name = self.config.get("optimizer", "adam")
        lr = self.config.get("learning_rate", 5e-4)
        weight_decay = self.config.get("weight_decay", 1e-4)

        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Learning rate scheduler
        scheduler_type = self.config.get("lr_scheduler", "plateau")
        if scheduler_type == "cosine":
            # Cosine annealing with warmup
            warmup_epochs = self.config.get("warmup_epochs", 5)
            total_epochs = self.config.get("epochs", 50)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
            )
            self.warmup_epochs = warmup_epochs
            self.use_warmup = True
            logger.info(f"Using Cosine Annealing LR with {warmup_epochs} warmup epochs")
        else:
            # Default: ReduceLROnPlateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=3,
            )
            self.use_warmup = False
            logger.info("Using ReduceLROnPlateau scheduler")

        logger.info(f"Optimizer: {optimizer_name}, LR: {lr}")

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]"
        )

        for batch in progress_bar:
            # Move batch to device
            user_ids = batch["user_id"].to(self.device)
            movie_ids = batch["movie_id"].to(self.device)
            ratings = batch["rating"].to(self.device)

            user_features = (
                batch["user_features"].to(self.device)
                if "user_features" in batch
                else None
            )
            movie_features = (
                batch["movie_features"].to(self.device)
                if "movie_features" in batch
                else None
            )

            # Forward pass
            self.optimizer.zero_grad()
            pred_ratings, user_emb, movie_emb = self.model(
                user_ids, movie_ids, user_features, movie_features
            )

            # Compute loss based on loss type
            loss_type = self.config.get("loss_type", "mse")
            use_regularization = self.config.get("use_regularization", False)

            if loss_type == "bpr":
                # Pure BPRLoss - use in-batch negatives (no need to sample!)
                # BPRLoss will automatically use other items in batch as negatives
                loss = self.criterion(user_emb, movie_emb, None)
            elif loss_type == "combined" or use_regularization:
                # For combined/regularized losses, still need explicit negatives
                neg_movie_emb = None
                if "neg_movie_ids" in batch:
                    neg_movie_ids = batch["neg_movie_ids"].to(self.device)
                    batch_size, n_negatives = neg_movie_ids.shape
                    neg_movie_ids_flat = neg_movie_ids.reshape(-1)
                    neg_movie_emb_flat = self.model.get_movie_embedding(
                        neg_movie_ids_flat, None
                    )
                    neg_movie_emb = neg_movie_emb_flat.reshape(
                        batch_size, n_negatives, -1
                    )
                loss = self.criterion(
                    pred_ratings, ratings, user_emb, movie_emb, neg_movie_emb
                )
            else:
                # MSELoss
                loss = self.criterion(pred_ratings, ratings)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.get("max_grad_norm", 1.0)
            )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            n_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on test set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        progress_bar = tqdm(
            self.test_loader, desc=f"Epoch {self.current_epoch + 1} [Val]"
        )

        for batch in progress_bar:
            # Move batch to device
            user_ids = batch["user_id"].to(self.device)
            movie_ids = batch["movie_id"].to(self.device)
            ratings = batch["rating"].to(self.device)

            user_features = (
                batch["user_features"].to(self.device)
                if "user_features" in batch
                else None
            )
            movie_features = (
                batch["movie_features"].to(self.device)
                if "movie_features" in batch
                else None
            )

            # Forward pass
            pred_ratings, user_emb, movie_emb = self.model(
                user_ids, movie_ids, user_features, movie_features
            )

            # Compute loss - use MSE for validation even if training uses BPR
            # (test set doesn't have negative samples)
            loss = torch.nn.functional.mse_loss(pred_ratings, ratings)

            # Update metrics
            total_loss += loss.item()
            n_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches

        return {"val_loss": avg_loss}

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        if is_best:
            best_path = path.parent / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def train(self):
        """Run full training loop with early stopping."""
        epochs = self.config.get("epochs", 50)
        patience = self.config.get("early_stopping_patience", 5)
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "models/checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Early stopping patience: {patience}")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics["val_loss"]

            # Log to MLflow
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )

            # Learning rate scheduling
            if hasattr(self, "use_warmup") and self.use_warmup:
                # Cosine annealing: step every epoch (after warmup)
                if epoch >= self.warmup_epochs:
                    self.scheduler.step()
            else:
                # ReduceLROnPlateau: step based on validation loss
                self.scheduler.step(val_loss)

            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info(f"✓ New best model! Val Loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            self.save_checkpoint(checkpoint_path, is_best=is_best)

            # Early stopping
            if self.patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(patience: {patience})"
                )
                break

        logger.info("\nTraining complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        # Log best model to MLflow
        best_model_path = checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            mlflow.pytorch.log_model(self.model, "model")
            logger.info("Logged best model to MLflow")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Two-Tower model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config JSON file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="two-tower-recommender",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config)

        # Create trainer
        trainer = Trainer(config)

        # Train
        trainer.train()

        # Log final metrics
        mlflow.log_metric("best_val_loss", trainer.best_val_loss)


if __name__ == "__main__":
    main()
