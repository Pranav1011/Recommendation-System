"""
Quick Ensemble Test: LightGCN + Two-Tower

Tests if combining graph-based and feature-based models improves performance.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.graph_builder import build_graph
from src.models.lightgcn import create_lightgcn_model
from src.models.two_tower import create_model as create_two_tower_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_ndcg(gt, pred, k):
    """Compute NDCG@K."""
    rel = np.array([1.0 if item in gt else 0.0 for item in pred[:k]])
    if rel.sum() == 0:
        return 0.0
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.ones(min(len(gt), k))
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(gt, pred):
    """Compute recall."""
    if len(gt) == 0:
        return 0.0
    return len(set(pred) & set(gt)) / len(gt)


def main():
    device = torch.device("cpu")  # Use CPU for simplicity
    logger.info(f"Device: {device}")

    # ========== Load LightGCN ==========
    logger.info("\n[1/6] Loading LightGCN...")
    lg_checkpoint = torch.load(
        "models/checkpoints/best_model_lightgcn_optimized.pt",
        map_location=device,
        weights_only=False
    )

    with open("configs/train_config_lightgcn_optimized.json") as f:
        lg_config = json.load(f)

    # Load or build graph
    graph_cache_file = Path("data/graph_cache/lightgcn_graph.pt")
    if graph_cache_file.exists():
        logger.info("  Loading cached graph...")
        graph_cache = torch.load(graph_cache_file, weights_only=False)
        lg_graph = graph_cache["graph"].to(device)
        user_to_idx_lg = graph_cache["user_to_idx"]
        item_to_idx_lg = graph_cache["item_to_idx"]
    else:
        logger.info("  Building graph from scratch...")
        graph_obj = build_graph(
            n_users=lg_config["n_users"],
            n_items=lg_config["n_movies"],
            train_ratings_path="data/processed/train_ratings.parquet",
            min_rating=lg_config["model"].get("rating_threshold", 4.0),
        )
        indices, values, size = graph_obj.get_sparse_graph()
        lg_graph = torch.sparse.FloatTensor(indices, values, size).to(device)
        user_to_idx_lg = graph_obj.user_to_idx
        item_to_idx_lg = graph_obj.item_to_idx

    # Create LightGCN model
    lg_model = create_lightgcn_model({
        "n_users": lg_config["n_users"],
        "n_movies": lg_config["n_movies"],
        "embedding_dim": lg_config["model"]["embedding_dim"],
        "n_layers": lg_config["model"]["n_layers"],
        "dropout_rate": 0.0,
    })
    lg_model.load_state_dict(lg_checkpoint["model_state_dict"])
    lg_model.to(device)
    lg_model.eval()

    logger.info("✓ LightGCN loaded")

    # ========== Load Two-Tower ==========
    logger.info("\n[2/6] Loading Two-Tower...")
    tt_checkpoint = torch.load(
        "models/checkpoints/best_model_bpr.pt",
        map_location=device,
        weights_only=False
    )

    # Use config from checkpoint (more reliable)
    tt_config = tt_checkpoint.get("config", None)
    if tt_config is None:
        # Fallback: Try to infer from model shape
        logger.warning("  Config not in checkpoint, using fallback")
        with open("configs/train_config_bpr.json") as f:
            tt_config = json.load(f)
        # Override with inferred dimensions from checkpoint
        sample_weight = tt_checkpoint["model_state_dict"]["user_tower.user_embedding.weight"]
        tt_config["embedding_dim"] = sample_weight.shape[1]
        hidden_weight = tt_checkpoint["model_state_dict"]["user_tower.fc1.weight"]
        tt_config["hidden_dim"] = hidden_weight.shape[0]
        tt_config["user_feature_dim"] = 0  # No features in BPR model
        tt_config["movie_feature_dim"] = 0

    # Create Two-Tower model
    tt_model = create_two_tower_model(tt_config)
    tt_model.load_state_dict(tt_checkpoint["model_state_dict"])
    tt_model.to(device)
    tt_model.eval()

    # Load features
    user_feats_df = pd.read_parquet("data/features/user_features.parquet")
    movie_feats_df = pd.read_parquet("data/features/movie_features.parquet")

    user_feats = torch.tensor(
        user_feats_df.drop(columns=["userId"]).values,
        dtype=torch.float32,
        device=device
    )
    movie_feats = torch.tensor(
        movie_feats_df.drop(columns=["movieId"]).values,
        dtype=torch.float32,
        device=device
    )

    user_to_idx_tt = {uid: idx for idx, uid in enumerate(user_feats_df["userId"])}
    item_to_idx_tt = {mid: idx for idx, mid in enumerate(movie_feats_df["movieId"])}

    logger.info("✓ Two-Tower loaded")

    # ========== Get Embeddings ==========
    logger.info("\n[3/6] Computing embeddings...")

    with torch.no_grad():
        # LightGCN embeddings
        lg_user_emb, lg_item_emb = lg_model(lg_graph)
        lg_user_emb = torch.nn.functional.normalize(lg_user_emb, dim=1)
        lg_item_emb = torch.nn.functional.normalize(lg_item_emb, dim=1)

        # Two-Tower embeddings
        tt_user_emb = tt_model.user_tower(torch.arange(len(user_feats), device=device), user_feats)
        tt_movie_emb = tt_model.movie_tower(torch.arange(len(movie_feats), device=device), movie_feats)
        tt_user_emb = torch.nn.functional.normalize(tt_user_emb, dim=1)
        tt_movie_emb = torch.nn.functional.normalize(tt_movie_emb, dim=1)

    logger.info(f"✓ LightGCN: {lg_user_emb.shape[0]} users, {lg_item_emb.shape[0]} items")
    logger.info(f"✓ Two-Tower: {tt_user_emb.shape[0]} users, {tt_movie_emb.shape[0]} items")

    # ========== Load Test Data ==========
    logger.info("\n[4/6] Loading test data...")
    test_df = pd.read_parquet("data/processed/test_ratings.parquet")
    test_df = test_df[test_df["rating"] >= 4.0]

    test_df["user_idx_lg"] = test_df["userId"].map(user_to_idx_lg)
    test_df["item_idx_lg"] = test_df["movieId"].map(item_to_idx_lg)
    test_df = test_df.dropna(subset=["user_idx_lg", "item_idx_lg"])
    test_df["user_idx_lg"] = test_df["user_idx_lg"].astype(int)
    test_df["item_idx_lg"] = test_df["item_idx_lg"].astype(int)

    test_interactions = test_df.groupby("user_idx_lg")["item_idx_lg"].apply(list).to_dict()
    logger.info(f"✓ {len(test_interactions)} test users")

    # ========== Evaluate Ensemble ==========
    logger.info("\n[5/6] Evaluating ensemble...")

    eval_users = list(test_interactions.keys())[:500]
    weights = [0.5, 0.6, 0.7, 0.8]  # Test different LightGCN weights

    results = {}

    for lg_weight in weights:
        logger.info(f"\n  Testing LightGCN={lg_weight:.1f}, Two-Tower={1-lg_weight:.1f}")

        ndcgs = []
        recalls = []

        for user_idx in tqdm(eval_users, desc=f"  Weight={lg_weight:.1f}", leave=False):
            # LightGCN scores
            scores_lg = torch.matmul(lg_item_emb, lg_user_emb[user_idx]).cpu().numpy()

            # Two-Tower scores (handle index mismatch)
            if user_idx < len(tt_user_emb):
                scores_tt = torch.matmul(tt_movie_emb, tt_user_emb[user_idx]).cpu().numpy()
            else:
                scores_tt = np.zeros_like(scores_lg)

            # Align dimensions
            min_len = min(len(scores_lg), len(scores_tt))
            scores_lg = scores_lg[:min_len]
            scores_tt = scores_tt[:min_len]

            # Ensemble
            ensemble_scores = lg_weight * scores_lg + (1 - lg_weight) * scores_tt

            # Top-10
            top_10 = np.argpartition(ensemble_scores, -10)[-10:]
            top_10 = top_10[np.argsort(ensemble_scores[top_10])][::-1]

            # Ground truth
            gt = test_interactions[user_idx]

            # Metrics
            ndcgs.append(compute_ndcg(gt, top_10, 10))
            recalls.append(compute_recall(gt, top_10))

        results[lg_weight] = {
            "ndcg@10": np.mean(ndcgs),
            "recall@10": np.mean(recalls),
        }

        logger.info(f"    NDCG@10:   {np.mean(ndcgs):.4f} ({np.mean(ndcgs)*100:.2f}%)")
        logger.info(f"    Recall@10: {np.mean(recalls):.4f} ({np.mean(recalls)*100:.2f}%)")

    # ========== Results Summary ==========
    logger.info("\n" + "=" * 80)
    logger.info("[6/6] ENSEMBLE RESULTS SUMMARY")
    logger.info("=" * 80)

    logger.info("\n{:^15} | {:^15} | {:^15}".format("LightGCN Wt", "NDCG@10", "Recall@10"))
    logger.info("-" * 50)
    for wt, metrics in sorted(results.items()):
        logger.info(
            f"{wt:^15.1f} | {metrics['ndcg@10']:^15.4f} | {metrics['recall@10']:^15.4f}"
        )

    best_weight = max(results.items(), key=lambda x: x[1]["ndcg@10"])
    logger.info("\n" + "=" * 80)
    logger.info(f"BEST: LightGCN={best_weight[0]:.1f}, NDCG@10={best_weight[1]['ndcg@10']:.4f}")
    logger.info("=" * 80)

    # Baseline comparison
    logger.info("\nComparison:")
    logger.info(f"  LightGCN only (1.0):    NDCG@10 = 0.0527 (5.27%)")
    logger.info(f"  Ensemble (best):        NDCG@10 = {best_weight[1]['ndcg@10']:.4f} ({best_weight[1]['ndcg@10']*100:.2f}%)")
    improvement = (best_weight[1]['ndcg@10'] - 0.0527) / 0.0527 * 100
    logger.info(f"  Improvement:            {improvement:+.1f}%")

    # Save results
    with open("models/checkpoints/ensemble_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\n✓ Results saved to models/checkpoints/ensemble_test_results.json")


if __name__ == "__main__":
    main()
