"""API schema definitions using Pydantic."""

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Single search result item."""

    id: str = Field(..., description="Unique identifier for the result (filename)")
    description: str = Field(..., description="Description of the image")
    seed: str = Field(..., description="Seed or rank identifier")
    relevance_reason: str | None = Field(
        None, description="Explanation of why this result is relevant"
    )
    score: float = Field(
        ..., description="Similarity score (0-1, higher is more similar)"
    )
    image_path: str = Field(..., description="Full path to the image file on disk")
    image_data: str = Field(..., description="Base64 encoded image data")


class SearchRequest(BaseModel):
    query: str | None = Field(None, description="Text query for searching images")
    image: str | None = Field(
        None, description="Base64 encoded image or image URL for similarity search"
    )
    liked_items: list[SearchResult] = Field(
        default_factory=list,
        description="Previously liked items to refine search",
    )
    disliked_items: list[SearchResult] = Field(
        default_factory=list,
        description="Previously disliked items to refine search",
    )
    positive_text: list[str] | None = Field(
        None, description="Positive text for refining search"
    )
    negative_text: list[str] | None = Field(
        None, description="Negative text for refining search"
    )

    k: int | None = Field(
        default=12,
        ge=1,
        le=100,
        description="Number of results to return (1-100)",
    )


class SearchResponse(BaseModel):
    """Response containing search results."""

    results: list[SearchResult] = Field(
        ..., description="List of search results ordered by relevance"
    )
    total: int | None = Field(None, description="Total number of results found")
    query: str | None = Field(None, description="Original query text")


class IndexStats(BaseModel):
    """Statistics about the search index."""

    num_images: int = Field(..., description="Number of images in the index")
    index_size: int = Field(..., description="Size of the FAISS index")
    embedding_dim: int = Field(..., description="Dimension of embedding vectors")
    model_name: str = Field(..., description="Name of the CLIP model used")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    index_folder: str = Field(..., description="Path to the index folder")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status (healthy/unhealthy)")
    search_engine: str = Field(
        ..., description="Search engine availability (available/unavailable)"
    )
    uptime: float | None = Field(None, description="Server uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str = Field(..., description="Error message")
    type: str | None = Field(None, description="Error type")
    code: str | None = Field(None, description="Error code")
