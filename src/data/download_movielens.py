"""
MovieLens Dataset Downloader

Downloads and extracts MovieLens datasets for the recommendation system.
Supports both 25M (full) and 1M (testing) datasets.
"""

import argparse
import logging
import zipfile
from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dataset URLs and metadata
DATASETS = {
    "25m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "zip_name": "ml-25m.zip",
        "extracted_folder": "ml-25m",
        "expected_files": ["ratings.csv", "movies.csv", "links.csv", "tags.csv"],
        "size_mb": 250,
    },
    "1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "zip_name": "ml-1m.zip",
        "extracted_folder": "ml-1m",
        "expected_files": ["ratings.dat", "movies.dat", "users.dat"],
        "size_mb": 6,
    },
    "latest-small": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "zip_name": "ml-latest-small.zip",
        "extracted_folder": "ml-latest-small",
        "expected_files": ["ratings.csv", "movies.csv", "links.csv", "tags.csv"],
        "size_mb": 1,
    },
}


def download_progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """
    Display download progress.

    Args:
        block_num: Current block number
        block_size: Size of each block in bytes
        total_size: Total file size in bytes
    """
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded / total_size) * 100)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        logger.info(
            f"Downloaded: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)"
        )


def download_dataset(
    size: Literal["25m", "1m", "latest-small"], output_dir: Path
) -> Path:
    """
    Download MovieLens dataset.

    Args:
        size: Dataset size to download (25m, 1m, or latest-small)
        output_dir: Directory to save the downloaded file

    Returns:
        Path to the downloaded zip file

    Raises:
        ValueError: If invalid size specified
        Exception: If download fails
    """
    if size not in DATASETS:
        raise ValueError(
            f"Invalid dataset size: {size}. Must be one of {list(DATASETS.keys())}"
        )

    dataset_info = DATASETS[size]
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / dataset_info["zip_name"]

    # Check if already downloaded
    if zip_path.exists():
        logger.info(f"Dataset already exists at {zip_path}")
        return zip_path

    # Download
    logger.info(
        f"Downloading MovieLens {size} dataset (~{dataset_info['size_mb']}MB)..."
    )
    logger.info(f"URL: {dataset_info['url']}")

    try:
        urlretrieve(dataset_info["url"], zip_path, reporthook=download_progress_hook)
        logger.info(f"Download complete: {zip_path}")
        return zip_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if zip_path.exists():
            zip_path.unlink()  # Clean up partial download
        raise


def extract_dataset(zip_path: Path, output_dir: Path, size: str) -> Path:
    """
    Extract the downloaded dataset.

    Args:
        zip_path: Path to the zip file
        output_dir: Directory to extract to
        size: Dataset size (for validation)

    Returns:
        Path to the extracted folder

    Raises:
        ValueError: If extraction fails or files are missing
    """
    dataset_info = DATASETS[size]
    extract_path = output_dir / dataset_info["extracted_folder"]

    # Check if already extracted
    if extract_path.exists():
        logger.info(f"Dataset already extracted at {extract_path}")
        return extract_path

    logger.info(f"Extracting {zip_path}...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        logger.info(f"Extraction complete: {extract_path}")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

    # Validate extracted files
    validate_dataset(extract_path, size)

    return extract_path


def validate_dataset(dataset_path: Path, size: str) -> None:
    """
    Validate that all expected files exist.

    Args:
        dataset_path: Path to extracted dataset
        size: Dataset size (for expected files)

    Raises:
        ValueError: If required files are missing
    """
    dataset_info = DATASETS[size]
    expected_files = dataset_info["expected_files"]

    logger.info("Validating dataset files...")
    missing_files = []

    for filename in expected_files:
        file_path = dataset_path / filename
        if not file_path.exists():
            missing_files.append(filename)
        else:
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"✓ {filename} ({file_size:.1f} MB)")

    if missing_files:
        raise ValueError(f"Missing required files: {missing_files}")

    logger.info("✓ All required files present")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download and extract MovieLens dataset"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["25m", "1m", "latest-small"],
        default="25m",
        help="Dataset size to download (default: 25m)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only download, skip extraction",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    try:
        # Download
        zip_path = download_dataset(args.size, output_dir)

        # Extract
        if not args.skip_extract:
            extract_path = extract_dataset(zip_path, output_dir, args.size)
            logger.info(f"\n✓ Dataset ready at: {extract_path}")
        else:
            logger.info(f"\n✓ Dataset downloaded to: {zip_path}")

        logger.info("\nNext steps:")
        logger.info("  1. Run data processing: python src/data/processor.py")
        logger.info("  2. Explore data: jupyter notebook notebooks/01_eda.ipynb")

    except Exception as e:
        logger.error(f"\n✗ Failed to download dataset: {e}")
        exit(1)


if __name__ == "__main__":
    main()
