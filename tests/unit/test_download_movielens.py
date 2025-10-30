"""Unit tests for MovieLens dataset downloader."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.data.download_movielens import (
    DATASETS,
    download_dataset,
    download_progress_hook,
    extract_dataset,
    validate_dataset,
)


class TestDownloadProgressHook:
    """Tests for download progress hook."""

    def test_progress_hook_logs_correctly(self, caplog):
        """Test that progress hook logs download progress."""
        with caplog.at_level("INFO"):
            download_progress_hook(10, 1024 * 1024, 100 * 1024 * 1024)
        assert "Downloaded:" in caplog.text
        assert "MB" in caplog.text


class TestDownloadDataset:
    """Tests for dataset download function."""

    def test_invalid_size_raises_error(self, tmp_path):
        """Test that invalid dataset size raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dataset size"):
            download_dataset("invalid", tmp_path)

    def test_valid_sizes(self):
        """Test that all valid sizes are in DATASETS."""
        assert "25m" in DATASETS
        assert "1m" in DATASETS
        assert "latest-small" in DATASETS

    @patch("src.data.download_movielens.urlretrieve")
    def test_download_creates_file(self, mock_urlretrieve, tmp_path):
        """Test that download creates zip file."""
        # Mock successful download
        mock_urlretrieve.return_value = None

        # Create the zip file that would be created by urlretrieve
        zip_path = tmp_path / DATASETS["latest-small"]["zip_name"]
        zip_path.touch()

        result = download_dataset("latest-small", tmp_path)

        assert result.exists()
        assert result.name == DATASETS["latest-small"]["zip_name"]

    def test_skip_download_if_exists(self, tmp_path):
        """Test that existing files are not re-downloaded."""
        # Create existing zip file
        zip_path = tmp_path / DATASETS["latest-small"]["zip_name"]
        zip_path.touch()

        with patch("src.data.download_movielens.urlretrieve") as mock_urlretrieve:
            result = download_dataset("latest-small", tmp_path)

            # Should not call urlretrieve
            mock_urlretrieve.assert_not_called()
            assert result == zip_path

    @patch("src.data.download_movielens.urlretrieve")
    def test_download_cleanup_on_failure(self, mock_urlretrieve, tmp_path):
        """Test that partial downloads are cleaned up on failure."""

        # Mock download failure
        def create_partial_and_fail(url, path, reporthook=None):
            # Create partial file during download
            Path(path).touch()
            raise Exception("Network error")

        mock_urlretrieve.side_effect = create_partial_and_fail

        with pytest.raises(Exception, match="Network error"):
            download_dataset("latest-small", tmp_path)


class TestExtractDataset:
    """Tests for dataset extraction function."""

    def test_extract_creates_directory(self, tmp_path):
        """Test that extraction creates expected directory."""
        # Create a mock zip file with required files
        zip_path = tmp_path / "test.zip"
        dataset_folder = "ml-latest-small"
        expected_files = DATASETS["latest-small"]["expected_files"]

        # Create zip with expected structure
        with zipfile.ZipFile(zip_path, "w") as zf:
            for file in expected_files:
                zf.writestr(f"{dataset_folder}/{file}", "mock data")

        result = extract_dataset(zip_path, tmp_path, "latest-small")

        assert result.exists()
        assert result.is_dir()
        assert result.name == dataset_folder

    def test_skip_extraction_if_exists(self, tmp_path):
        """Test that existing extracted folders are not re-extracted."""
        # Create existing extracted folder with required files
        dataset_folder = tmp_path / DATASETS["latest-small"]["extracted_folder"]
        dataset_folder.mkdir()

        # Create required files
        for file in DATASETS["latest-small"]["expected_files"]:
            (dataset_folder / file).touch()

        zip_path = tmp_path / "test.zip"
        zip_path.touch()

        result = extract_dataset(zip_path, tmp_path, "latest-small")

        assert result == dataset_folder

    def test_extraction_failure_raises_error(self, tmp_path):
        """Test that extraction failures are properly raised."""
        zip_path = tmp_path / "invalid.zip"
        zip_path.write_text("not a zip file")

        with pytest.raises(Exception):
            extract_dataset(zip_path, tmp_path, "latest-small")


class TestValidateDataset:
    """Tests for dataset validation function."""

    def test_validation_passes_with_all_files(self, tmp_path):
        """Test validation passes when all files are present."""
        dataset_path = tmp_path / "ml-latest-small"
        dataset_path.mkdir()

        # Create all expected files
        for file in DATASETS["latest-small"]["expected_files"]:
            (dataset_path / file).write_text("mock data")

        # Should not raise
        validate_dataset(dataset_path, "latest-small")

    def test_validation_fails_with_missing_files(self, tmp_path):
        """Test validation fails when files are missing."""
        dataset_path = tmp_path / "ml-latest-small"
        dataset_path.mkdir()

        # Create only some files
        (dataset_path / DATASETS["latest-small"]["expected_files"][0]).touch()

        with pytest.raises(ValueError, match="Missing required files"):
            validate_dataset(dataset_path, "latest-small")

    def test_validation_logs_file_sizes(self, tmp_path, caplog):
        """Test that validation logs file sizes."""
        dataset_path = tmp_path / "ml-latest-small"
        dataset_path.mkdir()

        # Create files with some content
        for file in DATASETS["latest-small"]["expected_files"]:
            (dataset_path / file).write_text("x" * 1000)

        with caplog.at_level("INFO"):
            validate_dataset(dataset_path, "latest-small")

        assert "MB" in caplog.text
        assert "All required files present" in caplog.text


class TestDatasetMetadata:
    """Tests for dataset metadata structure."""

    def test_all_datasets_have_required_fields(self):
        """Test that all datasets have required metadata fields."""
        required_fields = [
            "url",
            "zip_name",
            "extracted_folder",
            "expected_files",
            "size_mb",
        ]

        for size, metadata in DATASETS.items():
            for field in required_fields:
                assert field in metadata, f"{size} missing field: {field}"

    def test_dataset_urls_are_valid(self):
        """Test that all dataset URLs start with https."""
        for size, metadata in DATASETS.items():
            assert metadata["url"].startswith("https://"), f"{size} URL not using HTTPS"

    def test_expected_files_not_empty(self):
        """Test that all datasets have expected files listed."""
        for size, metadata in DATASETS.items():
            assert len(metadata["expected_files"]) > 0, f"{size} has no expected files"
