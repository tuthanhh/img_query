"""API routes for image search endpoints."""

import base64
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, HTTPException
from PIL import Image

from ..schema import IndexStats, SearchRequest, SearchResponse, SearchResult
from ..search import SearchEngine
from ..utils import get_logger

logger = get_logger()
router = APIRouter(prefix="/api", tags=["search"])

# Global search engine instance (will be set by main app)
_search_engine: SearchEngine | None = None


def set_search_engine(engine: SearchEngine | None):
    """Set the global search engine instance."""
    global _search_engine
    _search_engine = engine


def get_search_engine() -> SearchEngine:
    """Get the search engine instance or raise error if not initialized."""
    if _search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized. Please create an index first.",
        )
    return _search_engine


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Unified search endpoint supporting both JSON and form-data requests.

    Supports two formats:
    1. JSON (application/json): Send {"query": "text", "k": 12}

    Args:
        request: FastAPI request object

    Returns:
        SearchResponse with matching image results
    """
    engine = get_search_engine()

    # Extract request parameters
    search_query = request.query
    image_data = request.image
    num_results = request.k
    algorithm_type = request.algorithm_type
    relevance = [
        Image.open(item.image_path).convert("RGB") for item in request.liked_items
    ]
    irrelevance = [
        Image.open(item.image_path).convert("RGB") for item in request.disliked_items
    ]
    pos_text = request.positive_text
    neg_text = request.negative_text

    if not search_query and not image_data:
        raise HTTPException(
            status_code=400,
            detail="Either 'query' or 'image' must be provided",
        )

    if search_query:
        logger.info(
            f"Text search (JSON): '{search_query}' (k={num_results}, algo={algorithm_type})"
        )
        raw_results = engine.search_by_text(
            search_query,
            k=num_results if num_results else 12,
            relevance=relevance,
            irrelevance=irrelevance,
            positive_text=pos_text,
            negative_text=neg_text,
            algorithm_type=algorithm_type,
        )
        results = _format_search_results(raw_results)

        logger.info(f"Found {len(results)} results")
        return SearchResponse(
            results=results,
            total=len(results),
            query=search_query,
        )

    elif image_data:
        logger.info(
            f"Image search (base64): {image_data[:100]}' (k={num_results}, algo={algorithm_type})"
        )
        # convert image
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        raw_results = engine.search_by_image(
            image,
            k=num_results if num_results else 12,
            relevance=relevance,
            irrelevance=irrelevance,
            positive_text=pos_text,
            negative_text=neg_text,
            algorithm_type=algorithm_type,
        )
        results = _format_search_results(raw_results)

        logger.info(f"Found {len(results)} results")
        return SearchResponse(
            results=results,
            total=len(results),
            query=image_data,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'query' or 'image' must be provided",
        )


@router.get("/stats", response_model=IndexStats)
async def get_statistics():
    """
    Get statistics about the search index.

    Returns:
        IndexStats with information about the index
    """
    engine = get_search_engine()

    try:
        stats = engine.get_stats()
        return IndexStats(**stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}",
        ) from e


def _format_search_results(raw_results: list[dict]) -> list[SearchResult]:
    """
    Convert raw search results to API format.

    Args:
        raw_results: List of raw search results from SearchEngine

    Returns:
        List of SearchResult objects
    """
    results = []
    for r in raw_results:
        # Extract filename and create result
        filepath = Path(r["file"])
        filename = filepath.name
        stem = filepath.stem
        image_data = base64.b64encode(filepath.read_bytes()).decode("utf-8")

        results.append(
            SearchResult(
                id=stem,
                description=f"Image: {filename}",
                seed=str(r["rank"]),
                relevance_reason=f"Similarity score: {r['score']:.4f}",
                score=r["score"],
                image_path=str(filepath),
                image_data=image_data,
            )
        )

    return results
