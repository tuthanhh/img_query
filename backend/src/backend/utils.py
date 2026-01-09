"""Utility functions for the image retrieval system."""

import logging
import sys
from pathlib import Path
from typing import Optional

from PIL import Image


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional file path to write logs to
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logger
    logger = logging.getLogger("image_rel")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "image_rel") -> logging.Logger:
    """Get or create a logger instance."""
    return logging.getLogger(name)


def validate_image_path(image_path: str | Path) -> Path:
    """
    Validate that an image path exists and is a valid image file.

    Args:
        image_path: Path to the image file

    Returns:
        Path object if valid

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid image
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Try to open as image to validate
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Invalid image file: {path}. Error: {e}")

    return path


def validate_directory(dir_path: str | Path, create: bool = False) -> Path:
    """
    Validate that a directory exists.

    Args:
        dir_path: Path to the directory
        create: Whether to create the directory if it doesn't exist

    Returns:
        Path object if valid

    Raises:
        FileNotFoundError: If the directory doesn't exist and create=False
        ValueError: If the path exists but is not a directory
    """
    path = Path(dir_path)

    if not path.exists():
        if create:
            path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory not found: {path}")

    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    return path


def get_image_files(
    directory: str | Path,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".gif"),
    recursive: bool = False,
) -> list[Path]:
    """
    Get all image files from a directory.

    Args:
        directory: Directory to search
        extensions: Tuple of valid image extensions
        recursive: Whether to search recursively

    Returns:
        List of image file paths
    """
    dir_path = Path(directory)
    validate_directory(dir_path)

    image_files = []

    if recursive:
        for ext in extensions:
            image_files.extend(dir_path.rglob(f"*{ext}"))
            image_files.extend(dir_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            image_files.extend(dir_path.glob(f"*{ext}"))
            image_files.extend(dir_path.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def format_size(size_bytes: float) -> str:
    """
    Format byte size to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_time(seconds: float) -> str:
    """
    Format time duration to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1m 30s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def safe_load_image(image_path: str | Path) -> Image.Image | None:
    """
    Safely load an image with error handling.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object if successful, None otherwise
    """
    try:
        path = Path(image_path)
        img = Image.open(path)
        img.load()  # Force load to catch truncated images
        return img.convert("RGB")  # Ensure RGB format
    except Exception as e:
        logger = get_logger()
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None


def chunks(lst: list, n: int):
    """
    Yield successive n-sized chunks from a list.

    Args:
        lst: List to chunk
        n: Chunk size

    Yields:
        Chunks of the list
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def ensure_directory_structure(base_path: Path) -> dict:
    """
    Ensure the required directory structure exists.

    Args:
        base_path: Base path for the project

    Returns:
        Dictionary with created directory paths
    """
    directories = {
        "archive": base_path / "data" / "archive",
        "train": base_path / "data" / "train",
        "test": base_path / "data" / "test",
        "index": base_path / "data" / "index",
    }

    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print a separator line."""
    print(char * length)


def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    print_separator(char)
    print(text.center(80))
    print_separator(char)
