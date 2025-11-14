#!/usr/bin/env python3
"""
Setup Qdrant Vector Database

This script:
1. Loads user and movie embeddings from disk
2. Creates Qdrant collections
3. Indexes all embeddings with metadata
4. Verifies indexing was successful
5. Prints statistics

Usage:
    python scripts/setup_qdrant.py [--embeddings-dir data/embeddings] [--host localhost]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store.qdrant_client import QdrantManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_embeddings(embeddings_dir: Path, movies_file: Optional[Path] = None) -> tuple:
    """
    Load embeddings and metadata from disk.

    Args:
        embeddings_dir: Directory containing embeddings
        movies_file: Optional path to movies metadata file

    Returns:
        Tuple of (user_embeddings, movie_embeddings, movie_metadata, metadata)
    """
    logger.info(f"Loading embeddings from {embeddings_dir}")

    # Load user embeddings
    user_emb_path = embeddings_dir / "user_embeddings.npy"
    if not user_emb_path.exists():
        raise FileNotFoundError(f"User embeddings not found: {user_emb_path}")

    user_embeddings = np.load(user_emb_path)
    logger.info(f"Loaded user embeddings: {user_embeddings.shape}")

    # Load movie embeddings (try both naming conventions)
    movie_emb_path = embeddings_dir / "movie_embeddings.npy"
    if not movie_emb_path.exists():
        movie_emb_path = embeddings_dir / "item_embeddings.npy"

    if not movie_emb_path.exists():
        raise FileNotFoundError(f"Movie/item embeddings not found in {embeddings_dir}")

    movie_embeddings = np.load(movie_emb_path)
    logger.info(f"Loaded movie embeddings: {movie_embeddings.shape}")

    # Load embedding metadata (try both names)
    metadata_path = embeddings_dir / "embedding_metadata.json"
    if not metadata_path.exists():
        metadata_path = embeddings_dir / "embedding_statistics.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        logger.info(f"Loaded embedding metadata")
    else:
        logger.warning(f"Embedding metadata not found: {metadata_path}")

    # Load movie metadata
    if movies_file and movies_file.exists():
        movie_metadata = pd.read_parquet(movies_file)
        logger.info(f"Loaded movie metadata from {movies_file}: {movie_metadata.shape}")
    else:
        # Try default locations
        movie_features_path = Path("data/features/movie_features.parquet")
        if movie_features_path.exists():
            movie_metadata = pd.read_parquet(movie_features_path)
            logger.info(f"Loaded movie metadata: {movie_metadata.shape}")
        else:
            # Fallback to basic movies.parquet
            movie_basic_path = Path("data/processed/movies.parquet")
            if movie_basic_path.exists():
                movie_metadata = pd.read_parquet(movie_basic_path)
                logger.info(f"Loaded basic movie metadata: {movie_metadata.shape}")
            else:
                raise FileNotFoundError(
                    f"Movie metadata not found in {movie_features_path} or {movie_basic_path}"
                )

    return user_embeddings, movie_embeddings, movie_metadata, metadata


def verify_indexing(qdrant: QdrantManager, n_users: int, n_movies: int) -> bool:
    """
    Verify that indexing was successful.

    Args:
        qdrant: QdrantManager instance
        n_users: Expected number of users
        n_movies: Expected number of movies

    Returns:
        True if verification passed
    """
    logger.info("Verifying indexing...")

    # Check user collection
    user_info = qdrant.get_collection_info(QdrantManager.USER_COLLECTION)
    user_count = user_info.get("points_count", 0)

    if user_count != n_users:
        logger.error(
            f"User collection verification failed: "
            f"expected {n_users}, got {user_count}"
        )
        return False

    logger.info(f"User collection verified: {user_count:,} points")

    # Check movie collection
    movie_info = qdrant.get_collection_info(QdrantManager.MOVIE_COLLECTION)
    movie_count = movie_info.get("points_count", 0)

    if movie_count != n_movies:
        logger.error(
            f"Movie collection verification failed: "
            f"expected {n_movies}, got {movie_count}"
        )
        return False

    logger.info(f"Movie collection verified: {movie_count:,} points")

    # Test search functionality
    logger.info("Testing search functionality...")

    # Get a random user embedding
    test_user_id = np.random.randint(0, n_users)
    test_embedding = qdrant.get_user_embedding(test_user_id)

    if test_embedding is None:
        logger.error(f"Failed to retrieve test user {test_user_id}")
        return False

    # Search for similar movies
    results = qdrant.search_similar_movies(test_embedding, k=10)

    if len(results) != 10:
        logger.error(f"Search returned {len(results)} results, expected 10")
        return False

    logger.info(f"Search test passed: returned {len(results)} results")

    # Print sample results
    logger.info("\nSample recommendations for user {}:".format(test_user_id))
    for i, (movie_id, score, metadata) in enumerate(results[:5], 1):
        title = metadata.get("title", "Unknown")
        genres = metadata.get("genres", [])
        logger.info(f"  {i}. {title} (score: {score:.4f}, genres: {genres})")

    return True


def print_statistics(
    qdrant: QdrantManager,
    user_embeddings: np.ndarray,
    movie_embeddings: np.ndarray,
    metadata: dict,
) -> None:
    """Print indexing statistics."""
    logger.info("\n" + "=" * 80)
    logger.info("QDRANT INDEXING STATISTICS")
    logger.info("=" * 80)

    # Embedding statistics
    logger.info(f"\nEmbeddings:")
    logger.info(f"  Users:  {user_embeddings.shape[0]:,} x {user_embeddings.shape[1]}D")
    logger.info(f"  Movies: {movie_embeddings.shape[0]:,} x {movie_embeddings.shape[1]}D")

    # Collection statistics
    user_info = qdrant.get_collection_info(QdrantManager.USER_COLLECTION)
    movie_info = qdrant.get_collection_info(QdrantManager.MOVIE_COLLECTION)

    logger.info(f"\nUser Collection:")
    logger.info(f"  Points: {user_info.get('points_count', 0):,}")
    logger.info(f"  Indexed: {user_info.get('indexed_vectors_count', 0):,}")
    logger.info(f"  Segments: {user_info.get('segments_count', 0)}")
    logger.info(f"  Status: {user_info.get('status', 'unknown')}")

    logger.info(f"\nMovie Collection:")
    logger.info(f"  Points: {movie_info.get('points_count', 0):,}")
    logger.info(f"  Indexed: {movie_info.get('indexed_vectors_count', 0):,}")
    logger.info(f"  Segments: {movie_info.get('segments_count', 0)}")
    logger.info(f"  Status: {movie_info.get('status', 'unknown')}")

    # Model information
    if metadata:
        logger.info(f"\nModel Information:")
        logger.info(f"  Source: {metadata.get('model_path', 'unknown')}")
        if "n_users" in metadata:
            logger.info(f"  Training users: {metadata['n_users']:,}")
        if "n_movies" in metadata:
            logger.info(f"  Training movies: {metadata['n_movies']:,}")

    logger.info("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Setup Qdrant vector database")
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="data/embeddings",
        help="Directory containing embeddings",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Qdrant host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collections if they exist",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for indexing",
    )
    parser.add_argument(
        "--movies-file",
        type=str,
        default=None,
        help="Path to movies metadata parquet file",
    )

    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    movies_file = Path(args.movies_file) if args.movies_file else None

    try:
        # Load embeddings
        start_time = time.time()
        user_embeddings, movie_embeddings, movie_metadata, metadata = load_embeddings(
            embeddings_dir, movies_file
        )

        # Connect to Qdrant
        logger.info(f"\nConnecting to Qdrant at {args.host}:{args.port}")
        qdrant = QdrantManager(host=args.host, port=args.port)

        # Health check
        if not qdrant.health_check():
            logger.error("Qdrant health check failed")
            sys.exit(1)

        logger.info("Qdrant is healthy")

        # Create collections
        embedding_dim = user_embeddings.shape[1]
        logger.info(f"\nCreating collections (dim={embedding_dim})...")
        qdrant.create_collections(
            embedding_dim=embedding_dim, recreate=args.recreate
        )

        # Index user embeddings
        logger.info("\nIndexing user embeddings...")
        qdrant.index_user_embeddings(
            user_embeddings=user_embeddings, batch_size=args.batch_size
        )

        # Index movie embeddings
        logger.info("\nIndexing movie embeddings...")
        qdrant.index_movie_embeddings(
            movie_embeddings=movie_embeddings,
            movie_metadata=movie_metadata,
            batch_size=args.batch_size,
        )

        # Verify indexing
        logger.info("\nVerifying indexing...")
        if not verify_indexing(
            qdrant, len(user_embeddings), len(movie_embeddings)
        ):
            logger.error("Indexing verification failed")
            sys.exit(1)

        # Print statistics
        elapsed = time.time() - start_time
        print_statistics(qdrant, user_embeddings, movie_embeddings, metadata)

        logger.info(f"\nTotal time: {elapsed:.2f}s")
        logger.info("\nQdrant setup complete!")

    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
