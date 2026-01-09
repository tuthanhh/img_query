"""Utility module for splitting image datasets into train and test sets."""

import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

from sklearn.model_selection import train_test_split

from .config import default_config
from .utils import get_image_files, get_logger, validate_directory


def split_image(
    image_folder: str | Path,
    train_folder: str | Path,
    test_folder: str | Path,
    ratio: float = 0.2,
    state: int = 42,
    copy: bool = False,
) -> Tuple[Path, Path]:
    """
    Split images from a source folder into train and test sets.

    Args:
        image_folder: Source folder containing images
        train_folder: Destination folder for training images
        test_folder: Destination folder for test images
        ratio: Proportion of images to use for testing (0.0 to 1.0)
        state: Random seed for reproducibility
        copy: If True, copy files instead of moving them

    Returns:
        Tuple of (train_folder_path, test_folder_path)

    Raises:
        FileNotFoundError: If the source folder doesn't exist
        ValueError: If ratio is not between 0 and 1, or no images found
    """
    logger = get_logger()

    # Validate inputs
    if not 0 < ratio < 1:
        raise ValueError(f"Ratio must be between 0 and 1, got {ratio}")

    # Convert to Path objects and validate
    image_folder = validate_directory(image_folder, create=False)
    train_folder = validate_directory(train_folder, create=True)
    test_folder = validate_directory(test_folder, create=True)

    logger.info(f"Splitting images from: {image_folder}")
    logger.info(f"Train folder: {train_folder}")
    logger.info(f"Test folder: {test_folder}")
    logger.info(f"Test ratio: {ratio}, Random seed: {state}")

    # Get all image files
    image_files = get_image_files(
        image_folder,
        extensions=default_config.processing.image_extensions,
        recursive=False,
    )

    if not image_files:
        raise ValueError(f"No images found in {image_folder}")

    logger.info(f"Found {len(image_files)} images")

    # Split the image paths
    train_files, test_files = train_test_split(
        image_files, test_size=ratio, random_state=state
    )

    logger.info(f"Split: {len(train_files)} train, {len(test_files)} test")

    # Move or copy files to train folder
    operation = "Copying" if copy else "Moving"
    logger.info(f"{operation} files to train folder...")
    for file_path in train_files:
        dest_path = train_folder / file_path.name
        try:
            if copy:
                shutil.copy2(file_path, dest_path)
            else:
                shutil.move(str(file_path), str(dest_path))
        except Exception as e:
            logger.error(f"Failed to {operation.lower()} {file_path}: {e}")

    # Move or copy files to test folder
    logger.info(f"{operation} files to test folder...")
    for file_path in test_files:
        dest_path = test_folder / file_path.name
        try:
            if copy:
                shutil.copy2(file_path, dest_path)
            else:
                shutil.move(str(file_path), str(dest_path))
        except Exception as e:
            logger.error(f"Failed to {operation.lower()} {file_path}: {e}")

    logger.info("Dataset split completed successfully!")
    logger.info(f"Train images: {len(list(train_folder.iterdir()))}")
    logger.info(f"Test images: {len(list(test_folder.iterdir()))}")

    return train_folder, test_folder


def main() -> None:
    """Main function for running split_image as a script."""
    # Setup logging
    from .utils import setup_logging

    setup_logging(level=logging.INFO)
    logger = get_logger()

    try:
        # Use configuration values
        config = default_config
        image_folder = config.data.archive_dir
        train_folder = config.data.train_dir
        test_folder = config.data.test_dir
        ratio = config.processing.test_split_ratio
        state = config.processing.random_seed

        logger.info("=" * 80)
        logger.info("IMAGE DATASET SPLITTER".center(80))
        logger.info("=" * 80)

        train_folder, test_folder = split_image(
            image_folder=image_folder,
            train_folder=train_folder,
            test_folder=test_folder,
            ratio=ratio,
            state=state,
            copy=False,  # Move files by default
        )

        logger.info("=" * 80)
        logger.info("COMPLETED SUCCESSFULLY".center(80))
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during dataset splitting: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
