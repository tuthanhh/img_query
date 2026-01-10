"""Module for searching images using CLIP embeddings and FAISS."""

import json
import logging
import time
from pathlib import Path
from typing import cast

import clip
import faiss
import numpy as np
import torch
from google import genai
from google.genai import types
from PIL import Image

from .config import Config, default_config
from .utils import (
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

        self.client = genai.Client(api_key=self.config.gemini.api_key)
        # Load FAISS index
        index_path = self.index_folder / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")

        self.logger.info(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(str(index_path))

        # Move index to GPU if available and configured
        if self.config.index.use_gpu and self.device.type == "cuda":
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                self.logger.info("FAISS index moved to GPU")
            except Exception as e:
                self.logger.warning(f"Failed to move index to GPU: {e}")

        # Load metadata
        metadata_path = self.index_folder / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at: {metadata_path}")

        self.logger.info(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, encoding="utf-8") as f:
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

    def enrich_prompt(self, inp: str) -> list[str]:
        if not inp or not inp.strip():
            return []

        if not self.client:
            return [s.strip() for s in inp.split(",") if s.strip()]

        # REFINED PROMPT WITH STRICT RULES
        prompt = f"""
            You are an AI assistant for a CLIP-based image search engine.

            User Input: "{inp}"

            ## Task
            Expand the user's input into a list of 5-10 visual keywords that describe the image's style, content, lighting, and composition.

            ## Rules
            1. **JSON Only:** You must output valid JSON containing a single key "keywords".
            2. Do NOT use markdown formatting (no ```json blocks).
            3. Do NOT add conversational text. Just the JSON.
            4. **Visual Focus:** Keywords must be visual (e.g., "sunset", "neon", "bokeh") not abstract (e.g., "hope", "future").
            5. **No Safety Blocks:** If the input seems sensitive (e.g., "war", "bikini"), do NOT refuse. Instead, return safe, neutral photographic terms (e.g., "photojournalism", "beach scene", "summer").
            6. **Length:** Keep individual keywords short (1-3 words).
            7. **Diversity:** Include a mix of:
               - Subject (e.g., "cat", "car")
               - Style (e.g., "cyberpunk", "vintage", "minimalist")
               - Lighting (e.g., "cinematic lighting", "golden hour")
               - Quality (e.g., "4k", "detailed", "sharp focus")
            """

        try:
            # Keep these strict safety settings to prevent mid-stream cuts
            safety_settings = [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]

            response = self.client.models.generate_content(
                model=self.config.gemini.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="text/plain",
                    max_output_tokens=1000,
                    safety_settings=safety_settings,
                ),
            )
            raw_text = response.text
            if "```json" in raw_text:
                raw_text = raw_text.replace("```json", "").replace("```", "")
            # Then parse
            data = json.loads(raw_text or "{}")
            keywords = data.get("keywords", [])

            return [k.strip() for k in keywords if isinstance(k, str) and k.strip()]

        except json.JSONDecodeError:
            self.logger.error(
                f"JSON Error for input '{inp}'. Raw output:\n{response.text or ''}"
            )
            return [s.strip() for s in inp.split(",") if s.strip()]
        except Exception as e:
            self.logger.warning(f"Enrichment failed: {e}")
            return [s.strip() for s in inp.split(",") if s.strip()]

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
            image_tensor = (
                cast(torch.Tensor, self.preprocess(image)).unsqueeze(0).to(self.device)
            )

        elif isinstance(image_input, Image.Image):
            # PIL Image object
            image_tensor = (
                cast(torch.Tensor, self.preprocess(image_input))
                .unsqueeze(0)
                .to(self.device)
            )

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

    def _enrich_and_encode_text(
        self, text: str, use_enrichment: bool = True
    ) -> np.ndarray:
        """
        Enrich a text query using Gemini and encode into multiple embeddings,
        then average them.

        Args:
            text: Text query to enrich and encode
            use_enrichment: Whether to use Gemini enrichment (default: True)

        Returns:
            Normalized averaged embedding vector
        """
        if not use_enrichment or not self.config.gemini.enable_enrichment:
            return self._encode_text(text)

        # Use enrich_prompt to expand the query
        enriched_keywords = self.enrich_prompt(text)

        if not enriched_keywords:
            # Fallback to original text if enrichment fails
            return self._encode_text(text)

        # Encode each enriched keyword
        all_features = [self._encode_text(text)]
        for keyword in enriched_keywords:
            features = self._encode_text(keyword)
            all_features.append(features)

        # Average all feature vectors
        if len(all_features) == 1:
            return all_features[0]

        avg_features = np.mean(np.vstack(all_features), axis=0, keepdims=True).astype(
            np.float32
        )

        # Re-normalize after averaging
        faiss.normalize_L2(avg_features)

        return avg_features

    def _encode_text_with_template(
        self, text: str, template: str = "a photo of {}"
    ) -> np.ndarray:
        """
        Encode text with a prompt template.

        Args:
            text: Text to encode
            template: Template string with {} placeholder (default: "a photo of {}")

        Returns:
            Normalized embedding vector
        """
        prompt = template.format(text)
        return self._encode_text(prompt)

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
        positive_text: list[str] | None = None,
        negative_text: list[str] | None = None,
        k: int = 10,
        algorithm_type: str = "standard",
    ) -> list[dict]:
        """
        Search for similar images using an input image.

        Args:
            image_input: Query image (path, PIL Image, or tensor)
            relevance: List of relevant images for feedback
            irrelevance: List of irrelevant images for feedback
            positive_text: List of positive text queries
            negative_text: List of negative text queries
            k: Number of results to return
            algorithm_type: Relevance feedback algorithm ('standard', 'ide_regular', 'ide_dec_hi')

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
            # Encode positive text with positive prompt template
            positive_text_features = None
            if positive_text:
                positive_text_features = [
                    self._encode_text_with_template(text, template="a photo of {}")
                    for text in positive_text
                ]

            # Encode negative text with negative prompt template
            negative_text_features = None
            if negative_text:
                negative_text_features = [
                    self._encode_text_with_template(
                        text, template="an image without {}"
                    )
                    for text in negative_text
                ]
            # Select refinement algorithm based on type
            self.logger.info(f"Using refinement algorithm: {algorithm_type}")
            if algorithm_type == "ide_dec_hi":
                query_features = self._refine_query_dec_hi(
                    query_features,
                    relevance_features,
                    irrelevance_features,
                    positive_text_features,
                    negative_text_features,
                )
            elif algorithm_type == "ide_regular":
                query_features = self._refine_query_ide_reg(
                    query_features,
                    relevance_features,
                    irrelevance_features,
                    positive_text_features,
                    negative_text_features,
                )
            else:  # standard
                query_features = self._refine_query_standard(
                    query_features,
                    relevance_features,
                    irrelevance_features,
                    positive_text_features,
                    negative_text_features,
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
        positive_text: list[str] | None = None,
        negative_text: list[str] | None = None,
        k: int = 10,
        algorithm_type: str = "standard",
    ) -> list[dict]:
        """
        Search for images using a text query.

        Args:
            text_input: Text description of desired images
            relevance: List of relevant images for feedback
            irrelevance: List of irrelevant images for feedback
            positive_text: List of positive text queries
            negative_text: List of negative text queries
            k: Number of results to return
            algorithm_type: Relevance feedback algorithm ('standard', 'ide_regular', 'ide_dec_hi')

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
            # Encode query text with enrichment
            query_features = self._enrich_and_encode_text(
                text_input.strip(), use_enrichment=True
            )

            # Encode relevance feedback
            relevance_features = (
                [self._encode_image(item) for item in relevance] if relevance else None
            )
            irrelevance_features = (
                [self._encode_image(item) for item in irrelevance]
                if irrelevance
                else None
            )

            # Encode positive text with positive prompt template
            positive_text_features = None
            if positive_text:
                positive_text_features = [
                    self._encode_text_with_template(text, template="a photo of {}")
                    for text in positive_text
                ]

            # Encode negative text with negative prompt template
            negative_text_features = None
            if negative_text:
                negative_text_features = [
                    self._encode_text_with_template(
                        text, template="an image without {}"
                    )
                    for text in negative_text
                ]
            # Select refinement algorithm based on type
            self.logger.info(f"Using refinement algorithm: {algorithm_type}")
            if algorithm_type == "ide_dec_hi":
                query_features = self._refine_query_dec_hi(
                    query_features,
                    relevance_features,
                    irrelevance_features,
                    positive_text_features,
                    negative_text_features,
                )
            elif algorithm_type == "ide_regular":
                query_features = self._refine_query_ide_reg(
                    query_features,
                    relevance_features,
                    irrelevance_features,
                    positive_text_features,
                    negative_text_features,
                )
            else:  # standard
                query_features = self._refine_query_standard(
                    query_features,
                    relevance_features,
                    irrelevance_features,
                    positive_text_features,
                    negative_text_features,
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

    def _refine_query_standard(
        self,
        query,
        relevance_features,
        irrelevance_features,
        positive_text_features,
        negative_text_features,
        alpha=1.5,
        beta=0.5,
        gamma=0.15,
        text_beta=0.4,
        text_gamma=0.15,
    ):
        """
        Refine a search query based on relevance and irrelevance using the Rocchio algorithm.
        Query refinement is performed using the standard Rocchio algorithm.
        Args:
            query (np.array): The original query vector (shape: (1, D) or (D,)).
            relevance_features (list[np.array]): List of relevant feature vectors.
            irrelevance_features (list[np.array]): List of irrelevant feature vectors.
            positive_text_features (list[np.array]): List of positive text feature vectors.
            negative_text_features (list[np.array]): List of negative text feature vectors.
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

        # 5. Add Mean of Positive Text Features (Positive Beta)
        if positive_text_features is not None and len(positive_text_features) > 0:
            positive_text_matrix = np.vstack(positive_text_features)
            positive_text_mean = np.mean(positive_text_matrix, axis=0)
            refined_query += text_beta * positive_text_mean

        # 6. Subtract Mean of Negative Text Features (Negative gamma)
        if negative_text_features is not None and len(negative_text_features) > 0:
            negative_text_matrix = np.vstack(negative_text_features)
            negative_text_mean = np.mean(negative_text_matrix, axis=0)
            refined_query -= text_gamma * negative_text_mean

        # 4. Normalize the resulting vector
        # Since you are using Cosine Similarity (IndexFlatIP with normalized vectors),
        # the refined query MUST be normalized back to unit length.
        norm = np.linalg.norm(refined_query)
        if norm > 0:
            refined_query /= norm

        # return refined_query
        return refined_query.astype(np.float32).reshape(1, -1)

    def _refine_query_ide_reg(
        self,
        query,
        relevance_features,
        irrelevance_features,
        positive_text_features,
        negative_text_features,
        alpha=1.5,
        beta=0.5,
        gamma=0.15,
        text_beta=0.4,
        text_gamma=0.15,
    ):
        """
        Refine a search query based on relevance and irrelevance using the Rocchio algorithm.
        Query refinement is performed using the ide version of the Rocchio algorithm.
        Args:
            query (np.array): The original query vector (shape: (1, D) or (D,)).
            relevance_features (list[np.array]): List of relevant feature vectors.
            irrelevance_features (list[np.array]): List of irrelevant feature vectors.
            positive_text_features (list[np.array]): List of positive text feature vectors.
            negative_text_features (list[np.array]): List of negative text feature vectors.
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
            relevant_sum = np.sum(relevant_matrix, axis=0)
            refined_query += beta * relevant_sum

        # 3. Subtract Mean of Irrelevant Vectors (Gamma)
        if irrelevance_features is not None and len(irrelevance_features) > 0:
            irrelevant_matrix = np.vstack(irrelevance_features)
            irrelevant_sum = np.sum(irrelevant_matrix, axis=0)
            refined_query -= gamma * irrelevant_sum

        # 4. Add Mean of Positive Text Features (Positive Beta)
        if positive_text_features is not None and len(positive_text_features) > 0:
            positive_text_matrix = np.vstack(positive_text_features)
            positive_text_sum = np.sum(positive_text_matrix, axis=0)
            refined_query += text_beta * positive_text_sum

        # 5. Subtract Mean of Negative Text Features (Negative gamma)
        if negative_text_features is not None and len(negative_text_features) > 0:
            negative_text_matrix = np.vstack(negative_text_features)
            negative_text_sum = np.sum(negative_text_matrix, axis=0)
            refined_query -= text_gamma * negative_text_sum

        # 4. Normalize the resulting vector
        # Since you are using Cosine Similarity (IndexFlatIP with normalized vectors),
        # the refined query MUST be normalized back to unit length.
        norm = np.linalg.norm(refined_query)
        if norm > 0:
            refined_query /= norm

        # return refined_query
        return refined_query.astype(np.float32).reshape(1, -1)

    def _refine_query_dec_hi(
        self,
        query,
        relevance_features,
        irrelevance_features,
        positive_text_features,
        negative_text_features,
        alpha=1.5,
        beta=0.5,
        gamma=0.15,
        text_beta=0.4,
        text_gamma=0.15,
    ):
        """
        Refine a search query using the Ide Dec-Hi (Decoupled High) algorithm.

        Ide Dec-Hi Logic:
        - Relevants: Add sum of ALL relevant vectors (Ide standard).
        - Irrelevants: Subtract ONLY the single 'highest' non-relevant vector
          (the one most similar to the current query).

        Args:
            query (np.array): The original query vector (shape: (1, D) or (D,)).
            relevance_features (list[np.array]): List of relevant feature vectors.
            irrelevance_features (list[np.array]): List of irrelevant feature vectors.
            positive_text_features (list[np.array]): List of positive text feature vectors.
            negative_text_features (list[np.array]): List of negative text feature vectors.
            alpha (float): Weight for original query.
            beta (float): Weight for positive feedback.
            gamma (float): Weight for negative feedback.

        Returns:
            np.array: The refined and normalized query vector.
        """

        # Ensure query is float32 and correct shape
        query_vec = query.astype(np.float32)
        # Flatten for easier dot product calculations if needed, though usually (1, D) is fine
        query_flat = query_vec.flatten()

        # 1. Start with the original query weighted by alpha
        refined_query = query_vec * alpha

        # 2. Add Sum of Relevant Vectors (Beta)
        # Ide algorithm uses SUM, not MEAN (Rocchio uses Mean)
        if relevance_features is not None and len(relevance_features) > 0:
            relevant_matrix = np.vstack(relevance_features)
            relevant_sum = np.sum(relevant_matrix, axis=0)
            refined_query += beta * relevant_sum

        # 3. Subtract Single Highest Irrelevant Vector (Gamma) -> [Ide Dec-Hi Specific]
        if irrelevance_features is not None and len(irrelevance_features) > 0:
            irrelevant_matrix = np.vstack(irrelevance_features)

            # Find the non-relevant vector closest to the query (The "Highest" non-relevant)
            # We calculate dot product between the original query and all irrelevant vectors
            scores = np.dot(irrelevant_matrix, query_flat)
            highest_ranked_index = np.argmax(scores)

            # Only subtract the single most similar negative vector
            highest_irrelevant_vec = irrelevant_matrix[highest_ranked_index]
            refined_query -= gamma * highest_irrelevant_vec

        # 4. Add Sum of Positive Text Features (Text Beta)
        if positive_text_features is not None and len(positive_text_features) > 0:
            positive_text_matrix = np.vstack(positive_text_features)
            positive_text_sum = np.sum(positive_text_matrix, axis=0)
            refined_query += text_beta * positive_text_sum

        # 5. Subtract Single Highest Negative Text Feature (Text Gamma) -> [Ide Dec-Hi Specific]
        # Applying the Dec-Hi logic to text negatives as well for consistency
        if negative_text_features is not None and len(negative_text_features) > 0:
            negative_text_matrix = np.vstack(negative_text_features)

            # Find the negative text vector closest to the query
            text_scores = np.dot(negative_text_matrix, query_flat)
            highest_ranked_text_index = np.argmax(text_scores)

            highest_negative_text_vec = negative_text_matrix[highest_ranked_text_index]
            refined_query -= text_gamma * highest_negative_text_vec

        # 6. Normalize the resulting vector
        norm = np.linalg.norm(refined_query)
        if norm > 0:
            refined_query /= norm

        return refined_query.astype(np.float32).reshape(1, -1)


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
