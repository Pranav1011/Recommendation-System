"""
Pytest configuration and shared fixtures
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return the test data directory"""
    return project_root / "tests" / "fixtures"


@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary"""
    return {
        "data": {
            "raw_path": "data/raw",
            "processed_path": "data/processed",
            "train_test_split": 0.8,
        },
        "model": {"type": "two_tower", "embedding_dim": 128},
        "training": {"batch_size": 2048, "learning_rate": 0.001, "epochs": 20},
    }


@pytest.fixture
def redis_config():
    """Return Redis configuration"""
    return {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", 6379)),
        "db": 0,
    }


@pytest.fixture
def qdrant_config():
    """Return Qdrant configuration"""
    return {
        "host": os.getenv("QDRANT_HOST", "localhost"),
        "port": int(os.getenv("QDRANT_PORT", 6333)),
        "collection_name": "test-recommendations",
    }


# Add more shared fixtures as needed
