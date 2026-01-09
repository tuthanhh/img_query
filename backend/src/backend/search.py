"""Module for searching images using CLIP embeddings and FAISS."""

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import clip
import faiss
import numpy as np
import torch
from PIL import Image

from .config import Config, default_config
from .utils import (
    format_time,
    get_logger,
    safe_load_image,
    validate_directory,
    validate_image_path,
)


class SearchEngine:
    """Search engine for finding similar images using CLIP and FAISS."""

    def __init__(
        self,
        index_folder: str | Path,
        config: Config | None = None,
    ):
        """
        Initialize the search engine.

        Args:
            index_folder: Folder containing the FAISS index and metadata
            config: Configuration object. If None, uses default config.

        Raises:
            FileNotFoundError: If index or metadata files don't exist
            ValueError: If index is invalid or corrupt
        """
        self.config = config or default_config
        self.logger = get_logger()
        self.index_folder = validate_directory(index_folder, create=False)

        # Determine device
        if self.config.model.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.model.device)

        self.logger.info(f"Initializing search engine on device: {self.device}")

        # Load CLIP model
        self.logger.info(f"Loading CLIP model: {self.config.model.name}")
        self.model, self.preprocess = clip.load(
            self.config.model.name, device=self.device
        )
        self.model.eval()

        # Load FAISS index
        index_path = self.index_folder / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")

        self.logger.info(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(str(index_path))

        # Move index to GPU if available and configured
        if self.config.index.use_gpu and self.device.type == "cuda":
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.logger.info("FAISS index moved to GPU")
            except Exception as e:
                self.logger.warning(f"Failed to move index to GPU: {e}")

        # Load metadata
        metadata_path = self.index_folder / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at: {metadata_path}")

        self.logger.info(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Handle both old and new metadata formats
        if isinstance(metadata, list):
            # Old format: just a list of paths
            self.image_paths = metadata
            self.num_images = len(metadata)
        elif isinstance(metadata, dict):
            # New format: dictionary with metadata
            self.image_paths = metadata.get("image_paths", [])
            self.num_images = metadata.get("num_images", len(self.image_paths))
        else:
            raise ValueError("Invalid metadata format")

        self.logger.info(f"Loaded {self.num_images} images in index")

        # Validate index and metadata consistency
        if self.index.ntotal != self.num_images:
            self.logger.warning(
                f"Index size ({self.index.ntotal}) doesn't match "
                f"metadata ({self.num_images})"
            )

    def _encode_image(
        self, image_input: str | Path | Image.Image | torch.Tensor
    ) -> np.ndarray:
        """
        Encode an image into an embedding vector.

        Args:
            image_input: Image as path, PIL Image, or tensor

        Returns:
            Normalized embedding vector

        Raises:
            ValueError: If input type is unsupported
        """
        # Handle different input types
        if isinstance(image_input, (str, Path)):
            # Load from file path
            image_path = validate_image_path(image_input)
            image = safe_load_image(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_input}")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        elif isinstance(image_input, Image.Image):
            # PIL Image object
            image_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)

        elif isinstance(image_input, torch.Tensor):
            # PyTorch tensor
            image_tensor = image_input.to(self.device)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

        else:
            raise ValueError(
                f"Unsupported input type: {type(image_input)}. "
                "Expected str, Path, PIL.Image, or torch.Tensor"
            )

        # Encode image
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features.cpu().numpy().astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(features)

        return features

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into an embedding vector.

        Args:
            text: Text query

        Returns:
            Normalized embedding vector
        """
        # Tokenize and encode text
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            features = self.model.encode_text(text_tokens)
            features = features.cpu().numpy().astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(features)

        return features

    def _search(self, query_features: np.ndarray, k: int) -> list[dict]:
        """
        Perform similarity search in the index.

        Args:
            query_features: Query embedding vector
            k: Number of results to return

        Returns:
            List of search results with file paths and scores
        """
        # Validate k
        k = min(k, self.config.search.max_k, self.index.ntotal)
        k = max(k, 1)

        # Search index
        distances, indices = self.index.search(query_features, k)

        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            # Check for valid index
            if idx == -1 or idx >= len(self.image_paths):
                continue

            score = float(distances[0][i])

            # Filter by similarity threshold
            if score < self.config.search.similarity_threshold:
                continue

            results.append(
                {
                    "file": self.image_paths[idx],
                    "score": score,
                    "rank": i + 1,
                }
            )

        return results

    def search_by_image(
        self,
        image_input: str | Path | Image.Image | torch.Tensor,
        relevance: list[Image.Image] | None = None,
        irrelevance: list[Image.Image] | None = None,
        k: int = 10,
    ) -> list[dict]:
        """
        Search for similar images using an input image.

        Args:
            image_input: Query image (path, PIL Image, or tensor)
            k: Number of results to return

        Returns:
            List of dictionaries with keys:
                - 'file': Path to similar image
                - 'score': Similarity score (0-1, higher is more similar)
                - 'rank': Result rank (1-indexed)

        Raises:
            ValueError: If image input is invalid
        """
        start_time = time.time()

        try:
            # Encode query image
            query_features = self._encode_image(image_input)
            relevance_features = (
                [self._encode_image(item) for item in relevance] if relevance else None
            )
            irrelevance_features = (
                [self._encode_image(item) for item in irrelevance]
                if irrelevance
                else None
            )
            query_features = self._refine_query(
                query_features, relevance_features, irrelevance_features
            )
            # Search
            results = self._search(query_features, k)

            elapsed = time.time() - start_time
            self.logger.info(
                f"Image search completed in {elapsed:.3f}s, found {len(results)} results"
            )

            return results

        except Exception as e:
            self.logger.error(f"Image search failed: {e}", exc_info=True)
            raise

    def search_by_text(
        self,
        text_input: str,
        relevance: list[Image.Image] | None = None,
        irrelevance: list[Image.Image] | None = None,
        k: int = 10,
    ) -> list[dict]:
        """
        Search for images using a text query.

        Args:
            text_input: Text description of desired images
            k: Number of results to return

        Returns:
            List of dictionaries with keys:
                - 'file': Path to similar image
                - 'score': Similarity score (0-1, higher is more similar)
                - 'rank': Result rank (1-indexed)

        Raises:
            ValueError: If text input is empty
        """
        if not text_input or not text_input.strip():
            raise ValueError("Text input cannot be empty")

        start_time = time.time()

        try:
            # Encode query text
            query_features = self._encode_text(text_input.strip())
            relevance_features = (
                [self._encode_image(item) for item in relevance] if relevance else None
            )
            irrelevance_features = (
                [self._encode_image(item) for item in irrelevance]
                if irrelevance
                else None
            )
            query_features = self._refine_query(
                query_features, relevance_features, irrelevance_features
            )

            # Search
            results = self._search(query_features, k)

            elapsed = time.time() - start_time
            self.logger.info(
                f"Text search completed in {elapsed:.3f}s, found {len(results)} results"
            )

            return results

        except Exception as e:
            self.logger.error(f"Text search failed: {e}", exc_info=True)
            raise

    def get_stats(self) -> dict:
        """
        Get statistics about the search index.

        Returns:
            Dictionary with index statistics
        """
        return {
            "num_images": self.num_images,
            "index_size": self.index.ntotal,
            "embedding_dim": self.index.d,
            "model_name": self.config.model.name,
            "device": str(self.device),
            "index_folder": str(self.index_folder),
        }

    def _refine_query(
        self,
        query,
        relevance_features,
        irrelevance_features,
        alpha=1.0,
        beta=0.75,
        gamma=0.15,
    ):
        """
        Refine a search query based on relevance and irrelevance using the Rocchio algorithm.

        Args:
            query (np.array): The original query vector (shape: (1, D) or (D,)).
            relevance_features (list[np.array]): List of relevant feature vectors.
            irrelevance_features (list[np.array]): List of irrelevant feature vectors.
            alpha (float): Weight for original query.
            beta (float): Weight for positive feedback.
            gamma (float): Weight for negative feedback.

        Returns:
            np.array: The refined and normalized query vector.
        """

        # 1. Start with the original query weighted by alpha
        # Ensure it's float32 to match FAISS/PyTorch expectations
        refined_query = query.astype(np.float32) * alpha

        # 2. Add Mean of Relevant Vectors (Beta)
        if relevance_features is not None and len(relevance_features) > 0:
            # Stack list into matrix (N, D) and calculate mean along axis 0
            relevant_matrix = np.vstack(relevance_features)
            relevant_mean = np.mean(relevant_matrix, axis=0)
            refined_query += beta * relevant_mean

        # 3. Subtract Mean of Irrelevant Vectors (Gamma)
        if irrelevance_features is not None and len(irrelevance_features) > 0:
            irrelevant_matrix = np.vstack(irrelevance_features)
            irrelevant_mean = np.mean(irrelevant_matrix, axis=0)
            refined_query -= gamma * irrelevant_mean

        # 4. Normalize the resulting vector
        # Since you are using Cosine Similarity (IndexFlatIP with normalized vectors),
        # the refined query MUST be normalized back to unit length.
        norm = np.linalg.norm(refined_query)
        if norm > 0:
            refined_query /= norm

        return refined_query


def main() -> None:
    """Main function for running search as an interactive script."""
    from .utils import print_header, setup_logging

    setup_logging(level=logging.INFO)
    logger = get_logger()

    try:
        # Use configuration values
        config = default_config
        index_folder = config.data.index_dir

        print_header("IMAGE SEARCH ENGINE")

        # Initialize search engine
        search_engine = SearchEngine(index_folder, config)

        # Display statistics
        stats = search_engine.get_stats()
        print("\nIndex Statistics:")
        print(f"  - Images indexed: {stats['num_images']}")
        print(f"  - Model: {stats['model_name']}")
        print(f"  - Device: {stats['device']}")
        print(f"  - Index location: {stats['index_folder']}")

        # Interactive search loop
        while True:
            print("\n" + "=" * 80)
            print("\nSearch Options:")
            print("  1. Search by text")
            print("  2. Search by image")
            print("  3. Show statistics")
            print("  4. Exit")

            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == "1":
                # Text search
                text = input("\nEnter text description: ").strip()
                if not text:
                    print("Error: Text cannot be empty")
                    continue

                k = input(
                    f"Number of results (default: {config.search.default_k}): "
                ).strip()
                k = int(k) if k.isdigit() else config.search.default_k

                print(f"\nSearching for: '{text}'...")
                results = search_engine.search_by_text(text, k=k)

                print(f"\nFound {len(results)} results:\n")
                for result in results:
                    print(
                        f"  [{result['rank']}] {result['file']} (score: {result['score']:.4f})"
                    )

            elif choice == "2":
                # Image search
                image_path = input("\nEnter image path: ").strip()
                if not image_path:
                    print("Error: Image path cannot be empty")
                    continue

                if not Path(image_path).exists():
                    print(f"Error: File not found: {image_path}")
                    continue

                k = input(
                    f"Number of results (default: {config.search.default_k}): "
                ).strip()
                k = int(k) if k.isdigit() else config.search.default_k

                print(f"\nSearching for similar images to: {image_path}...")
                results = search_engine.search_by_image(image_path, k=k)

                print(f"\nFound {len(results)} results:\n")
                for result in results:
                    print(
                        f"  [{result['rank']}] {result['file']} (score: {result['score']:.4f})"
                    )

            elif choice == "3":
                # Show statistics
                print("\n" + "=" * 80)
                print("INDEX STATISTICS".center(80))
                print("=" * 80)
                for key, value in stats.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")

            elif choice == "4":
                # Exit
                print("\nGoodbye!")
                break

            else:
                print("Invalid choice. Please enter 1-4.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"Error in search engine: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
