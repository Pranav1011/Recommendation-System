"""
Ensemble Recommender System

Combines multiple recommendation models using weighted blending.
Supports different ensemble strategies:
- Weighted linear combination
- Score normalization
- Rank-based fusion
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EnsembleRecommender:
    """
    Ensemble multiple recommendation models.

    Combines predictions from different models using weighted averaging
    to balance quality, diversity, and specialized strengths.
    """

    def __init__(
        self,
        model_configs: Dict[str, Dict],
        ensemble_weights: Optional[Dict[str, float]] = None,
        combination_method: str = "weighted_score",
    ):
        """
        Initialize Ensemble Recommender.

        Args:
            model_configs: Dict mapping model names to config dicts
                Example: {
                    "rtx4090": {"embeddings_dir": "data/embeddings_optimized"},
                    "h200": {"embeddings_dir": "data/embeddings_h200"}
                }
            ensemble_weights: Dict mapping model names to weights (default: equal)
            combination_method: How to combine predictions
                - "weighted_score": Weighted average of similarity scores
                - "weighted_rank": Weighted average of ranks
                - "borda_count": Borda count voting
        """
        self.model_configs = model_configs
        self.combination_method = combination_method

        # Set weights (default: equal)
        if ensemble_weights is None:
            n_models = len(model_configs)
            self.weights = {name: 1.0 / n_models for name in model_configs.keys()}
        else:
            # Normalize weights to sum to 1
            total = sum(ensemble_weights.values())
            self.weights = {k: v / total for k, v in ensemble_weights.items()}

        logger.info(f"Initialized Ensemble with {len(model_configs)} models")
        logger.info(f"Weights: {self.weights}")
        logger.info(f"Combination method: {combination_method}")

        # Load embeddings for each model
        self.embeddings = {}
        self._load_all_embeddings()

    def _load_all_embeddings(self) -> None:
        """Load user and movie embeddings for all models."""
        for model_name, config in self.model_configs.items():
            logger.info(f"Loading embeddings for {model_name}...")
            embeddings_dir = Path(config["embeddings_dir"])

            user_emb_path = embeddings_dir / "user_embeddings.npy"
            movie_emb_path = embeddings_dir / "movie_embeddings.npy"

            if not user_emb_path.exists() or not movie_emb_path.exists():
                raise FileNotFoundError(
                    f"Embeddings not found for {model_name} in {embeddings_dir}"
                )

            user_emb = np.load(user_emb_path)
            movie_emb = np.load(movie_emb_path)

            self.embeddings[model_name] = {
                "user": user_emb,
                "movie": movie_emb,
            }

            logger.info(
                f"  {model_name}: {user_emb.shape[0]} users, "
                f"{movie_emb.shape[0]} movies, {user_emb.shape[1]}D embeddings"
            )

    def get_recommendations(
        self,
        user_indices: List[int],
        k: int = 100,
        batch_size: int = 1000,
        return_scores: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Get top-K recommendations for users using ensemble.

        Args:
            user_indices: List of user indices
            k: Number of recommendations per user
            batch_size: Batch size for processing
            return_scores: If True, also return similarity scores

        Returns:
            Tuple of (recommendations, scores)
                - recommendations: List of top-K movie indices per user
                - scores: Optional list of scores per recommendation
        """
        logger.info(
            f"Generating ensemble recommendations for {len(user_indices)} users "
            f"(K={k}, method={self.combination_method})"
        )

        if self.combination_method == "weighted_score":
            return self._weighted_score_combination(
                user_indices, k, batch_size, return_scores
            )
        elif self.combination_method == "weighted_rank":
            return self._weighted_rank_combination(
                user_indices, k, batch_size, return_scores
            )
        elif self.combination_method == "borda_count":
            return self._borda_count_combination(
                user_indices, k, batch_size, return_scores
            )
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

    def _weighted_score_combination(
        self,
        user_indices: List[int],
        k: int,
        batch_size: int,
        return_scores: bool,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Combine models using weighted average of similarity scores.

        Most common and effective method. Directly averages similarity scores
        from each model, weighted by model importance.
        """
        all_recommendations = []
        all_scores = [] if return_scores else None

        for i in tqdm(range(0, len(user_indices), batch_size), desc="Batches"):
            batch_user_indices = user_indices[i : i + batch_size]

            # Get number of movies from first model
            first_model = list(self.embeddings.keys())[0]
            n_movies = self.embeddings[first_model]["movie"].shape[0]

            # Initialize combined scores
            combined_scores = np.zeros((len(batch_user_indices), n_movies))

            # For each model, compute scores and add weighted contribution
            for model_name, weight in self.weights.items():
                # Get embeddings
                user_emb = self.embeddings[model_name]["user"][batch_user_indices]
                movie_emb = self.embeddings[model_name]["movie"]

                # Compute similarity scores (dot product since embeddings are normalized)
                scores = user_emb @ movie_emb.T  # (batch_size, n_movies)

                # Add weighted contribution
                combined_scores += weight * scores

            # Get top-K for each user
            top_k_indices = np.argsort(-combined_scores, axis=1)[:, :k]

            # Get corresponding scores if requested
            if return_scores:
                top_k_scores = np.take_along_axis(
                    combined_scores, top_k_indices, axis=1
                )
                all_scores.extend(top_k_scores.tolist())

            all_recommendations.extend(top_k_indices.tolist())

        return all_recommendations, all_scores

    def _weighted_rank_combination(
        self,
        user_indices: List[int],
        k: int,
        batch_size: int,
        return_scores: bool,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Combine models using weighted average of ranks.

        Useful when models have different score scales. Converts scores to
        ranks (1=best), then averages ranks with weights.
        """
        all_recommendations = []
        all_scores = [] if return_scores else None

        for i in tqdm(range(0, len(user_indices), batch_size), desc="Batches"):
            batch_user_indices = user_indices[i : i + batch_size]

            # Get number of movies from first model
            first_model = list(self.embeddings.keys())[0]
            n_movies = self.embeddings[first_model]["movie"].shape[0]

            # Initialize combined ranks
            combined_ranks = np.zeros((len(batch_user_indices), n_movies))

            # For each model, compute ranks and add weighted contribution
            for model_name, weight in self.weights.items():
                # Get embeddings
                user_emb = self.embeddings[model_name]["user"][batch_user_indices]
                movie_emb = self.embeddings[model_name]["movie"]

                # Compute similarity scores
                scores = user_emb @ movie_emb.T

                # Convert to ranks (lower rank = better)
                # argsort gives indices that would sort array (low to high)
                # argsort again gives rank of each element
                ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)

                # Add weighted contribution
                combined_ranks += weight * ranks

            # Get top-K items with lowest combined rank
            top_k_indices = np.argsort(combined_ranks, axis=1)[:, :k]

            # Compute scores for top-K if requested (use weighted score)
            if return_scores:
                # Recompute scores for top-K using weighted score method
                top_k_scores = np.zeros((len(batch_user_indices), k))
                for model_name, weight in self.weights.items():
                    user_emb = self.embeddings[model_name]["user"][batch_user_indices]
                    movie_emb = self.embeddings[model_name]["movie"]
                    scores = user_emb @ movie_emb.T
                    top_k_model_scores = np.take_along_axis(
                        scores, top_k_indices, axis=1
                    )
                    top_k_scores += weight * top_k_model_scores

                all_scores.extend(top_k_scores.tolist())

            all_recommendations.extend(top_k_indices.tolist())

        return all_recommendations, all_scores

    def _borda_count_combination(
        self,
        user_indices: List[int],
        k: int,
        batch_size: int,
        return_scores: bool,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Combine models using Borda count voting.

        Each model votes for items based on rank. Item gets n_movies - rank
        points. Sum points across models (weighted), select top-K.
        """
        all_recommendations = []
        all_scores = [] if return_scores else None

        for i in tqdm(range(0, len(user_indices), batch_size), desc="Batches"):
            batch_user_indices = user_indices[i : i + batch_size]

            # Get number of movies from first model
            first_model = list(self.embeddings.keys())[0]
            n_movies = self.embeddings[first_model]["movie"].shape[0]

            # Initialize Borda counts
            borda_counts = np.zeros((len(batch_user_indices), n_movies))

            # For each model, compute Borda counts and add weighted contribution
            for model_name, weight in self.weights.items():
                # Get embeddings
                user_emb = self.embeddings[model_name]["user"][batch_user_indices]
                movie_emb = self.embeddings[model_name]["movie"]

                # Compute similarity scores
                scores = user_emb @ movie_emb.T

                # Convert to ranks
                ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)

                # Borda count: item at rank r gets (n_movies - r - 1) points
                counts = n_movies - ranks - 1

                # Add weighted contribution
                borda_counts += weight * counts

            # Get top-K items with highest Borda count
            top_k_indices = np.argsort(-borda_counts, axis=1)[:, :k]

            # Get Borda counts as scores if requested
            if return_scores:
                top_k_borda = np.take_along_axis(borda_counts, top_k_indices, axis=1)
                all_scores.extend(top_k_borda.tolist())

            all_recommendations.extend(top_k_indices.tolist())

        return all_recommendations, all_scores

    def analyze_model_contributions(
        self, user_indices: List[int], k: int = 10
    ) -> Dict[str, Dict]:
        """
        Analyze how much each model contributes to final recommendations.

        Args:
            user_indices: Sample of users to analyze
            k: Number of recommendations to analyze

        Returns:
            Dictionary with contribution statistics per model
        """
        logger.info(f"Analyzing model contributions for {len(user_indices)} users...")

        # Get recommendations from ensemble
        ensemble_recs, _ = self.get_recommendations(user_indices, k=k)

        # Get recommendations from each individual model
        individual_recs = {}
        for model_name in self.embeddings.keys():
            recs = []
            for user_idx in user_indices:
                user_emb = self.embeddings[model_name]["user"][user_idx]
                movie_emb = self.embeddings[model_name]["movie"]
                scores = user_emb @ movie_emb.T
                top_k = np.argsort(-scores)[:k]
                recs.append(set(top_k))
            individual_recs[model_name] = recs

        # Compute overlap statistics
        contributions = {}
        for model_name in self.embeddings.keys():
            overlaps = []
            for i in range(len(user_indices)):
                ensemble_set = set(ensemble_recs[i])
                model_set = individual_recs[model_name][i]
                overlap = len(ensemble_set & model_set) / k
                overlaps.append(overlap)

            contributions[model_name] = {
                "mean_overlap": np.mean(overlaps),
                "std_overlap": np.std(overlaps),
                "min_overlap": np.min(overlaps),
                "max_overlap": np.max(overlaps),
                "weight": self.weights[model_name],
            }

        # Log results
        logger.info("\nModel Contributions:")
        for model_name, stats in contributions.items():
            logger.info(
                f"  {model_name}: {stats['mean_overlap']:.2%} overlap "
                f"(weight={stats['weight']:.2f})"
            )

        return contributions


def create_ensemble(
    model_dirs: Dict[str, str],
    weights: Optional[Dict[str, float]] = None,
    method: str = "weighted_score",
) -> EnsembleRecommender:
    """
    Create an ensemble recommender from model directories.

    Args:
        model_dirs: Dict mapping model names to embedding directories
        weights: Optional weights for each model
        method: Combination method

    Returns:
        EnsembleRecommender instance

    Example:
        >>> ensemble = create_ensemble({
        ...     "rtx4090": "data/embeddings_optimized",
        ...     "h200": "data/embeddings_h200"
        ... }, weights={"rtx4090": 0.6, "h200": 0.4})
    """
    configs = {
        name: {"embeddings_dir": dir_path} for name, dir_path in model_dirs.items()
    }
    return EnsembleRecommender(configs, weights, method)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create ensemble
    ensemble = create_ensemble(
        {
            "rtx4090": "data/embeddings_optimized",
            "h200": "data/embeddings_h200",
        },
        weights={"rtx4090": 0.6, "h200": 0.4},
        method="weighted_score",
    )

    # Get recommendations for first 10 users
    user_indices = list(range(10))
    recommendations, scores = ensemble.get_recommendations(
        user_indices, k=10, return_scores=True
    )

    print(f"\nGenerated {len(recommendations)} recommendation lists")
    print(f"Sample recommendation for user 0: {recommendations[0][:5]}")
    print(f"Sample scores: {scores[0][:5]}")

    # Analyze contributions
    contributions = ensemble.analyze_model_contributions(user_indices, k=10)
