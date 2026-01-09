"""Module for creating and managing FAISS indexes for image embeddings."""

import json
import logging
import time
from pathlib import Path
from typing import cast

import clip
import faiss
import numpy as np
import torch

from .config import Config, default_config
from .utils import (
    chunks,
    format_time,
    get_image_files,
    get_logger,
    safe_load_image,
    validate_directory,
)


class ImageIndexer:
    """Class for creating FAISS indexes from image embeddings."""

    def __init__(self, config: Config | None = None):
        """
        Initialize the ImageIndexer.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or default_config
        self.logger = get_logger()

        # Determine device
        if self.config.model.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.model.device

        self.logger.info(f"Using device: {self.device}")

        # Load CLIP model
        self.logger.info(f"Loading CLIP model: {self.config.model.name}")
        self.model, self.preprocess = clip.load(
            self.config.model.name, device=self.device
        )
        self.model.eval()  # Set to evaluation mode

    def encode_image(self, image_path: Path) -> np.ndarray | None:
        """
        Encode a single image to an embedding vector.

        Args:
            image_path: Path to the image file

        Returns:
            Normalized embedding vector or None if encoding fails
        """
        try:
            # Load and preprocess image
            image = safe_load_image(image_path)
            if image is None:
                return None

            image_tensor = (
                cast(torch.Tensor, self.preprocess(image)).unsqueeze(0).to(self.device)
            )

            # Encode image
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding.cpu().numpy().astype(np.float32)

            return embedding

        except Exception as e:
            self.logger.warning(f"Failed to encode {image_path}: {e}")
            return None

    def encode_images_batch(
        self, image_paths: list[Path], batch_size: int | None = None
    ) -> tuple[list[np.ndarray], list[Path]]:
        """
        Encode multiple images in batches.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing. If None, uses config value.

        Returns:
            Tuple of (embeddings list, successfully processed paths list)
        """
        if batch_size is None:
            batch_size = self.config.processing.batch_size

        embeddings = []
        successful_paths = []

        self.logger.info(
            f"Encoding {len(image_paths)} images in batches of {batch_size}"
        )

        # Process in batches
        for i, batch_paths in enumerate(chunks(image_paths, batch_size)):
            batch_images = []
            batch_valid_paths = []

            # Load and preprocess batch
            for img_path in batch_paths:
                img = safe_load_image(img_path)
                if img is not None:
                    batch_images.append(self.preprocess(img))
                    batch_valid_paths.append(img_path)

            if not batch_images:
                continue

            try:
                # Stack into batch tensor
                batch_tensor = torch.stack(batch_images).to(self.device)

                # Encode batch
                with torch.no_grad():
                    batch_embeddings = self.model.encode_image(batch_tensor)
                    batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float32)

                # Add to results
                for j, embedding in enumerate(batch_embeddings):
                    embeddings.append(embedding)
                    successful_paths.append(batch_valid_paths[j])

                # Progress update
                if (i + 1) % 10 == 0 or (i + 1) * batch_size >= len(image_paths):
                    processed = min((i + 1) * batch_size, len(image_paths))
                    self.logger.info(
                        f"Processed {processed}/{len(image_paths)} images "
                        f"({processed / len(image_paths) * 100:.1f}%)"
                    )

            except Exception as e:
                self.logger.error(f"Batch encoding failed: {e}")
                # Fall back to individual encoding
                for img_path in batch_valid_paths:
                    embedding = self.encode_image(img_path)
                    if embedding is not None:
                        embeddings.append(embedding)
                        successful_paths.append(img_path)

        return embeddings, successful_paths

    def create_faiss_index(self, embeddings: list[np.ndarray]) -> faiss.IndexFlatIP:
        """
        Create a FAISS index from embeddings.

        Args:
            embeddings: List of embedding vectors

        Returns:
            FAISS index
        """
        self.logger.info("Creating FAISS index...")

        # Concatenate embeddings
        embedding_matrix = np.vstack(embeddings).astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embedding_matrix)

        # Create index
        dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

        # Add vectors to index
        # Pyright sees the internal 'add(n, x)' signature, but Python handles 'n' automatically.
        index.add(embedding_matrix)  # pyright: ignore[reportCallIssue]

        self.logger.info(
            f"Index created with {index.ntotal} vectors of dimension {dimension}"
        )

        return index

    def save_index(
        self,
        index: faiss.Index,
        image_paths: list[Path],
        index_folder: Path,
    ) -> None:
        """
        Save FAISS index and metadata to disk.

        Args:
            index: FAISS index to save
            image_paths: List of image paths corresponding to index vectors
            index_folder: Folder to save index and metadata
        """
        index_folder = validate_directory(index_folder, create=True)

        # Save FAISS index
        index_path = index_folder / "index.faiss"
        faiss.write_index(index, str(index_path))
        self.logger.info(f"FAISS index saved to: {index_path}")

        # Save metadata
        metadata_path = index_folder / "metadata.json"
        metadata = {
            "image_paths": [str(path) for path in image_paths],
            "num_images": len(image_paths),
            "embedding_dim": self.config.model.embedding_dim,
            "model_name": self.config.model.name,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Metadata saved to: {metadata_path}")

    def create_index(
        self,
        image_folder: Path,
        index_folder: Path,
        batch_size: int | None = None,
    ) -> tuple[faiss.Index, list[Path]]:
        """
        Create index from images in a folder.

        Args:
            image_folder: Folder containing images
            index_folder: Folder to save the index
            batch_size: Batch size for processing

        Returns:
            Tuple of (FAISS index, list of indexed image paths)

        Raises:
            ValueError: If no valid images found
        """
        start_time = time.time()

        # Validate directories
        image_folder = validate_directory(image_folder, create=False)
        index_folder = validate_directory(index_folder, create=True)

        self.logger.info(f"Indexing images from: {image_folder}")

        # Get image files
        image_files = get_image_files(
            image_folder,
            extensions=self.config.processing.image_extensions,
            recursive=False,
        )

        if not image_files:
            raise ValueError(f"No images found in {image_folder}")

        self.logger.info(f"Found {len(image_files)} images to index")

        # Encode images
        embeddings, successful_paths = self.encode_images_batch(image_files, batch_size)

        if not embeddings:
            raise ValueError("Failed to encode any images")

        self.logger.info(
            f"Successfully encoded {len(embeddings)}/{len(image_files)} images"
        )

        # Create FAISS index
        index = self.create_faiss_index(embeddings)

        # Save index and metadata
        self.save_index(index, successful_paths, index_folder)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Indexing completed in {format_time(elapsed_time)}")
        self.logger.info(
            f"Average time per image: {elapsed_time / len(embeddings):.3f}s"
        )

        return index, successful_paths


def create_index(
    image_folder: str | Path,
    index_folder: str | Path,
    config: Config | None = None,
) -> None:
    """
    Convenience function to create an index from images.

    Args:
        image_folder: Folder containing images to index
        index_folder: Folder to save the index
        config: Configuration object. If None, uses default config.
    """
    indexer = ImageIndexer(config)
    indexer.create_index(Path(image_folder), Path(index_folder))


def main() -> None:
    """Main function for running indexer as a script."""
    from .utils import setup_logging

    setup_logging(level=logging.INFO)
    logger = get_logger()

    try:
        # Use configuration values
        config = default_config
        input_folder = config.data.train_dir
        index_folder = config.data.index_dir

        logger.info("=" * 80)
        logger.info("IMAGE INDEXER".center(80))
        logger.info("=" * 80)

        # Create indexer and build index
        indexer = ImageIndexer(config)
        indexer.create_index(input_folder, index_folder)

        logger.info("=" * 80)
        logger.info("INDEXING COMPLETED SUCCESSFULLY".center(80))
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
