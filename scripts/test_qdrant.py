#!/usr/bin/env python3
"""
Test Qdrant Vector Database

Comprehensive test suite for Qdrant deployment verification.

Usage:
    python scripts/test_qdrant.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store.qdrant_client import QdrantManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_health_check(qdrant: QdrantManager) -> bool:
    """Test Qdrant health check."""
    logger.info("Testing health check...")
    healthy = qdrant.health_check()

    if healthy:
        logger.info("✅ Health check passed")
        return True
    else:
        logger.error("❌ Health check failed")
        return False


def test_collection_info(qdrant: QdrantManager) -> bool:
    """Test collection info retrieval."""
    logger.info("Testing collection info...")

    user_info = qdrant.get_collection_info('user_embeddings')
    movie_info = qdrant.get_collection_info('movie_embeddings')

    # Expected counts
    expected_users = 162541
    expected_movies = 59047

    user_count = user_info.get('points_count', 0)
    movie_count = movie_info.get('points_count', 0)

    if user_count != expected_users:
        logger.error(f"❌ User count mismatch: expected {expected_users}, got {user_count}")
        return False

    if movie_count != expected_movies:
        logger.error(f"❌ Movie count mismatch: expected {expected_movies}, got {movie_count}")
        return False

    logger.info(f"✅ User collection: {user_count:,} points")
    logger.info(f"✅ Movie collection: {movie_count:,} points")
    return True


def test_user_retrieval(qdrant: QdrantManager) -> bool:
    """Test user embedding retrieval."""
    logger.info("Testing user embedding retrieval...")

    test_ids = [0, 100, 1000, 10000]

    for user_id in test_ids:
        embedding = qdrant.get_user_embedding(user_id)

        if embedding is None:
            logger.error(f"❌ Failed to retrieve user {user_id}")
            return False

        if embedding.shape != (192,):
            logger.error(f"❌ User {user_id} shape mismatch: expected (192,), got {embedding.shape}")
            return False

        if np.isnan(embedding).any():
            logger.error(f"❌ User {user_id} has NaN values")
            return False

    logger.info(f"✅ User retrieval passed for {len(test_ids)} users")
    return True


def test_movie_retrieval(qdrant: QdrantManager) -> bool:
    """Test movie embedding retrieval."""
    logger.info("Testing movie embedding retrieval...")

    test_ids = [0, 100, 1000, 10000]

    for movie_idx in test_ids:
        embedding = qdrant.get_movie_embedding(movie_idx)

        if embedding is None:
            logger.error(f"❌ Failed to retrieve movie {movie_idx}")
            return False

        if embedding.shape != (192,):
            logger.error(f"❌ Movie {movie_idx} shape mismatch: expected (192,), got {embedding.shape}")
            return False

        if np.isnan(embedding).any():
            logger.error(f"❌ Movie {movie_idx} has NaN values")
            return False

    logger.info(f"✅ Movie retrieval passed for {len(test_ids)} movies")
    return True


def test_search_basic(qdrant: QdrantManager) -> bool:
    """Test basic similarity search."""
    logger.info("Testing basic similarity search...")

    # Get a test user embedding
    user_emb = qdrant.get_user_embedding(100)

    if user_emb is None:
        logger.error("❌ Failed to get test user embedding")
        return False

    # Search for similar movies
    results = qdrant.search_similar_movies(user_emb, k=10)

    if len(results) != 10:
        logger.error(f"❌ Search returned {len(results)} results, expected 10")
        return False

    # Verify result format
    for movie_id, score, metadata in results:
        if not isinstance(movie_id, int):
            logger.error(f"❌ Invalid movie_id type: {type(movie_id)}")
            return False

        if not isinstance(score, float):
            logger.error(f"❌ Invalid score type: {type(score)}")
            return False

        if not isinstance(metadata, dict):
            logger.error(f"❌ Invalid metadata type: {type(metadata)}")
            return False

        if 'title' not in metadata:
            logger.error(f"❌ Missing title in metadata")
            return False

    # Verify scores are descending
    scores = [score for _, score, _ in results]
    if scores != sorted(scores, reverse=True):
        logger.error("❌ Scores are not in descending order")
        return False

    logger.info(f"✅ Basic search passed, returned {len(results)} results")
    logger.info(f"   Top result: {results[0][2]['title']} (score: {results[0][1]:.4f})")
    return True


def test_search_with_genre_filter(qdrant: QdrantManager) -> bool:
    """Test similarity search with genre filter."""
    logger.info("Testing search with genre filter...")

    user_emb = qdrant.get_user_embedding(100)

    # Test with Action genre
    results = qdrant.search_similar_movies(
        user_emb,
        k=10,
        genre_filter=['Action']
    )

    if len(results) == 0:
        logger.error("❌ Genre filter returned no results")
        return False

    # Verify all results have Action genre
    for movie_id, score, metadata in results:
        genres = metadata.get('genres', [])
        if 'Action' not in genres:
            logger.error(f"❌ Movie {metadata['title']} doesn't have Action genre: {genres}")
            return False

    logger.info(f"✅ Genre filter passed, returned {len(results)} Action movies")
    logger.info(f"   Top result: {results[0][2]['title']}")
    return True


def test_get_recommendations(qdrant: QdrantManager) -> bool:
    """Test get_recommendations method."""
    logger.info("Testing get_recommendations...")

    test_users = [0, 100, 1000]

    for user_id in test_users:
        recommendations = qdrant.get_recommendations(user_id, k=10)

        if len(recommendations) != 10:
            logger.error(f"❌ get_recommendations returned {len(recommendations)} results for user {user_id}")
            return False

    logger.info(f"✅ get_recommendations passed for {len(test_users)} users")
    return True


def test_search_performance(qdrant: QdrantManager, n_queries: int = 100) -> bool:
    """Test search performance."""
    logger.info(f"Testing search performance ({n_queries} queries)...")

    latencies = []

    for i in range(n_queries):
        user_id = i % 10000  # Cycle through first 10k users
        user_emb = qdrant.get_user_embedding(user_id)

        if user_emb is None:
            continue

        start = time.time()
        results = qdrant.search_similar_movies(user_emb, k=10)
        latency = (time.time() - start) * 1000  # Convert to ms

        latencies.append(latency)

    if not latencies:
        logger.error("❌ No successful queries")
        return False

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean = np.mean(latencies)

    logger.info(f"✅ Performance metrics:")
    logger.info(f"   Mean latency: {mean:.2f}ms")
    logger.info(f"   P50 latency: {p50:.2f}ms")
    logger.info(f"   P95 latency: {p95:.2f}ms")
    logger.info(f"   P99 latency: {p99:.2f}ms")
    logger.info(f"   Throughput: {1000 / mean:.1f} QPS")

    # Performance targets
    if p99 > 200:
        logger.warning(f"⚠️  P99 latency ({p99:.2f}ms) exceeds target (200ms)")

    return True


def test_edge_cases(qdrant: QdrantManager) -> bool:
    """Test edge cases."""
    logger.info("Testing edge cases...")

    # Test non-existent user
    invalid_user = qdrant.get_user_embedding(999999999)
    if invalid_user is not None:
        logger.error("❌ Retrieved embedding for non-existent user")
        return False

    # Test with k=1
    user_emb = qdrant.get_user_embedding(0)
    results = qdrant.search_similar_movies(user_emb, k=1)
    if len(results) != 1:
        logger.error(f"❌ k=1 returned {len(results)} results")
        return False

    # Test with k=100
    results = qdrant.search_similar_movies(user_emb, k=100)
    if len(results) != 100:
        logger.error(f"❌ k=100 returned {len(results)} results")
        return False

    logger.info("✅ Edge cases passed")
    return True


def print_summary(results: dict) -> None:
    """Print test summary."""
    total = len(results)
    passed = sum(1 for v in results.values() if v)

    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 80)
    logger.info(f"Total: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed!")
    else:
        logger.error(f"❌ {total - passed} tests failed")

    logger.info("=" * 80 + "\n")


def main():
    """Main entry point."""
    logger.info("Starting Qdrant test suite...\n")

    try:
        # Connect to Qdrant
        logger.info("Connecting to Qdrant...")
        qdrant = QdrantManager(host='localhost', port=6333)
        logger.info(f"Connected to Qdrant at localhost:6333\n")

        # Run tests
        results = {
            "Health Check": test_health_check(qdrant),
            "Collection Info": test_collection_info(qdrant),
            "User Retrieval": test_user_retrieval(qdrant),
            "Movie Retrieval": test_movie_retrieval(qdrant),
            "Basic Search": test_search_basic(qdrant),
            "Genre Filter": test_search_with_genre_filter(qdrant),
            "Get Recommendations": test_get_recommendations(qdrant),
            "Performance": test_search_performance(qdrant, n_queries=50),
            "Edge Cases": test_edge_cases(qdrant),
        }

        # Print summary
        print_summary(results)

        # Exit with appropriate code
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)

    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
