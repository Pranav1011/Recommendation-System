"""
Qdrant Vector Database Client

Production-grade client for managing embeddings and similarity search.
Supports:
- User and movie embeddings indexing
- Fast similarity search with HNSW index
- Metadata filtering (genre, year, rating)
- Batch operations for scalability
- Connection pooling and error handling
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manage Qdrant vector database for recommendation system."""

    # Collection names
    USER_COLLECTION = "user_embeddings"
    MOVIE_COLLECTION = "movie_embeddings"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        timeout: int = 30,
        prefer_grpc: bool = False,
    ):
        """
        Initialize Qdrant client.

        Args:
            host: Qdrant server host
            port: Qdrant server port (6333 for HTTP, 6334 for gRPC)
            timeout: Request timeout in seconds
            prefer_grpc: Use gRPC instead of HTTP
        """
        self.host = host
        self.port = port
        self.timeout = timeout

        # Create client
        self.client = QdrantClient(
            host=host,
            port=port,
            timeout=timeout,
            prefer_grpc=prefer_grpc,
        )

        logger.info(f"Connected to Qdrant at {host}:{port}")

    @classmethod
    def from_env(cls) -> "QdrantManager":
        """Create QdrantManager from environment variables."""
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        timeout = int(os.getenv("QDRANT_TIMEOUT", "30"))

        return cls(host=host, port=port, timeout=timeout)

    def create_collections(
        self,
        embedding_dim: int = 128,
        recreate: bool = False,
    ) -> None:
        """
        Create Qdrant collections for users and movies.

        Args:
            embedding_dim: Dimensionality of embeddings
            recreate: If True, delete and recreate existing collections
        """
        # HNSW index parameters for fast similarity search
        # M=16: Number of edges per node (higher = better recall, more memory)
        # ef_construct=100: Quality of index (higher = better quality, slower build)
        hnsw_config = models.HnswConfigDiff(
            m=16,
            ef_construct=100,
            full_scan_threshold=10000,
        )

        # Optimizers config for indexing performance
        optimizers_config = models.OptimizersConfigDiff(
            indexing_threshold=20000,  # Start indexing after 20k vectors
        )

        collections = [
            (self.USER_COLLECTION, "User embeddings for recommendation"),
            (self.MOVIE_COLLECTION, "Movie embeddings for recommendation"),
        ]

        for collection_name, description in collections:
            # Check if collection exists
            try:
                collection_info = self.client.get_collection(collection_name)
                if recreate:
                    logger.info(f"Deleting existing collection: {collection_name}")
                    self.client.delete_collection(collection_name)
                else:
                    logger.info(
                        f"Collection {collection_name} already exists "
                        f"with {collection_info.points_count} points"
                    )
                    continue
            except (UnexpectedResponse, ValueError):
                # Collection doesn't exist, will create
                pass

            # Create collection
            logger.info(f"Creating collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=models.Distance.COSINE,  # Cosine similarity
                    hnsw_config=hnsw_config,
                ),
                optimizers_config=optimizers_config,
            )

            logger.info(f"Created collection: {collection_name} ({embedding_dim}D)")

    def index_user_embeddings(
        self,
        user_embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> None:
        """
        Index user embeddings into Qdrant.

        Args:
            user_embeddings: User embedding matrix (n_users, embedding_dim)
            batch_size: Number of vectors to upload per batch
        """
        n_users, embedding_dim = user_embeddings.shape
        logger.info(f"Indexing {n_users:,} user embeddings ({embedding_dim}D)...")

        start_time = time.time()

        # Upload in batches for better performance
        for batch_start in range(0, n_users, batch_size):
            batch_end = min(batch_start + batch_size, n_users)
            batch_embeddings = user_embeddings[batch_start:batch_end]

            # Create points
            points = [
                models.PointStruct(
                    id=user_id,
                    vector=embedding.tolist(),
                    payload={"user_id": int(user_id)},
                )
                for user_id, embedding in enumerate(batch_embeddings, start=batch_start)
            ]

            # Upload batch
            self.client.upsert(
                collection_name=self.USER_COLLECTION,
                points=points,
                wait=False,  # Async upload for speed
            )

            if (batch_end % 10000) == 0 or batch_end == n_users:
                logger.info(f"  Indexed {batch_end:,}/{n_users:,} users")

        elapsed = time.time() - start_time
        logger.info(
            f"Indexed {n_users:,} user embeddings in {elapsed:.2f}s "
            f"({n_users/elapsed:.0f} vectors/sec)"
        )

    def index_movie_embeddings(
        self,
        movie_embeddings: np.ndarray,
        movie_metadata: pd.DataFrame,
        batch_size: int = 100,
    ) -> None:
        """
        Index movie embeddings with metadata into Qdrant.

        Args:
            movie_embeddings: Movie embedding matrix (n_movies, embedding_dim)
            movie_metadata: DataFrame with movie metadata (movieId, title, genres, etc.)
            batch_size: Number of vectors to upload per batch
        """
        n_movies, embedding_dim = movie_embeddings.shape
        logger.info(f"Indexing {n_movies:,} movie embeddings ({embedding_dim}D)...")

        start_time = time.time()

        # Upload in batches
        for batch_start in range(0, n_movies, batch_size):
            batch_end = min(batch_start + batch_size, n_movies)
            batch_embeddings = movie_embeddings[batch_start:batch_end]

            # Create points with metadata
            points = []
            for idx, embedding in enumerate(batch_embeddings, start=batch_start):
                # Get movie metadata
                movie_row = movie_metadata.iloc[idx]

                # Parse genres
                genres_str = movie_row.get("genres", "")
                genres = genres_str.split("|") if isinstance(genres_str, str) else []

                # Create payload with all metadata
                payload = {
                    "movie_id": int(movie_row["movieId"]),
                    "title": str(movie_row.get("title", "")),
                    "genres": genres,
                }

                # Add optional metadata if available
                if "year" in movie_row:
                    payload["year"] = int(movie_row["year"])
                if "avg_rating" in movie_row:
                    payload["avg_rating"] = float(movie_row["avg_rating"])
                if "popularity" in movie_row:
                    payload["popularity"] = int(movie_row["popularity"])

                points.append(
                    models.PointStruct(
                        id=idx,
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                )

            # Upload batch
            self.client.upsert(
                collection_name=self.MOVIE_COLLECTION,
                points=points,
                wait=False,
            )

            if (batch_end % 10000) == 0 or batch_end == n_movies:
                logger.info(f"  Indexed {batch_end:,}/{n_movies:,} movies")

        elapsed = time.time() - start_time
        logger.info(
            f"Indexed {n_movies:,} movie embeddings in {elapsed:.2f}s "
            f"({n_movies/elapsed:.0f} vectors/sec)"
        )

    def search_similar_movies(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        genre_filter: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_rating: Optional[float] = None,
    ) -> List[Tuple[int, float, Dict]]:
        """
        Search for similar movies using vector similarity.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            genre_filter: Filter by genres (OR condition)
            min_year: Minimum release year
            max_year: Maximum release year
            min_rating: Minimum average rating

        Returns:
            List of (movie_id, score, metadata) tuples
        """
        # Build filter conditions
        must_conditions = []

        if genre_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="genres",
                    match=models.MatchAny(any=genre_filter),
                )
            )

        if min_year is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(gte=min_year),
                )
            )

        if max_year is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(lte=max_year),
                )
            )

        if min_rating is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="avg_rating",
                    range=models.Range(gte=min_rating),
                )
            )

        # Create filter
        query_filter = None
        if must_conditions:
            query_filter = models.Filter(must=must_conditions)

        # Search
        results = self.client.search(
            collection_name=self.MOVIE_COLLECTION,
            query_vector=query_vector.tolist(),
            limit=k,
            query_filter=query_filter,
        )

        # Format results
        return [
            (
                result.payload["movie_id"],
                result.score,
                result.payload,
            )
            for result in results
        ]

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get user embedding by ID.

        Args:
            user_id: User ID

        Returns:
            User embedding vector or None if not found
        """
        try:
            points = self.client.retrieve(
                collection_name=self.USER_COLLECTION,
                ids=[user_id],
                with_vectors=True,
            )

            if points and points[0].vector is not None:
                return np.array(points[0].vector, dtype=np.float32)
            return None

        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}")
            return None

    def get_movie_embedding(self, movie_idx: int) -> Optional[np.ndarray]:
        """
        Get movie embedding by index.

        Args:
            movie_idx: Movie index in embedding matrix

        Returns:
            Movie embedding vector or None if not found
        """
        try:
            points = self.client.retrieve(
                collection_name=self.MOVIE_COLLECTION,
                ids=[movie_idx],
                with_vectors=True,
            )

            if points and points[0].vector is not None:
                return np.array(points[0].vector, dtype=np.float32)
            return None

        except Exception as e:
            logger.error(f"Error retrieving movie {movie_idx}: {e}")
            return None

    def get_recommendations(
        self,
        user_id: int,
        k: int = 10,
        genre_filter: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_rating: Optional[float] = None,
    ) -> List[Tuple[int, float, Dict]]:
        """
        Get movie recommendations for a user.

        Args:
            user_id: User ID
            k: Number of recommendations
            genre_filter: Filter by genres
            min_year: Minimum release year
            max_year: Maximum release year
            min_rating: Minimum average rating

        Returns:
            List of (movie_id, score, metadata) tuples
        """
        # Get user embedding
        user_embedding = self.get_user_embedding(user_id)

        if user_embedding is None:
            logger.warning(f"User {user_id} not found")
            return []

        # Search similar movies
        return self.search_similar_movies(
            query_vector=user_embedding,
            k=k,
            genre_filter=genre_filter,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
        )

    def get_collection_info(self, collection_name: str) -> Dict:
        """
        Get collection statistics.

        Args:
            collection_name: Name of collection

        Returns:
            Dictionary with collection info
        """
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "segments_count": info.segments_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def health_check(self) -> bool:
        """
        Check if Qdrant is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple health check - list collections
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    def close(self) -> None:
        """Close Qdrant client connection."""
        try:
            self.client.close()
            logger.info("Closed Qdrant connection")
        except Exception as e:
            logger.error(f"Error closing Qdrant connection: {e}")
