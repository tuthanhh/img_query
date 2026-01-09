"""FastAPI server for image search API."""

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router as api_router
from .api.routes import set_search_engine
from .config import default_config
from .schema import HealthResponse
from .search import SearchEngine
from .utils import get_logger, setup_logging

# Setup logging
setup_logging(level=logging.INFO)
logger = get_logger()

# Track server startup time
_startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global _startup_time
    _startup_time = time.time()

    # Startup
    try:
        config = default_config
        index_folder = config.data.index_dir

        if not index_folder.exists():
            logger.warning(
                f"Index folder not found at {index_folder}. "
                "Search functionality will be limited. "
                "Run 'backend index' to create an index."
            )
        else:
            logger.info("Initializing search engine...")
            search_engine = SearchEngine(index_folder, config)
            set_search_engine(search_engine)
            logger.info("Search engine initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}", exc_info=True)
        logger.warning("API will run in limited mode without search functionality")

    yield

    # Shutdown
    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Visual Search API",
    version="0.1.0",
    description="Image retrieval system using CLIP embeddings and FAISS",
    lifespan=lifespan,
)

# Configure CORS
config = default_config
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    from .api.routes import _search_engine

    return {
        "message": "Visual Search API",
        "version": "0.1.0",
        "status": "ready",
        "search_available": _search_engine is not None,
        "endpoints": {
            "health": "/health",
            "stats": "/api/stats",
            "search": "/api/search (supports both text query and image upload)",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    from .api.routes import _search_engine

    global _startup_time
    uptime = time.time() - _startup_time if _startup_time else None

    return HealthResponse(
        status="healthy",
        search_engine="available" if _search_engine else "unavailable",
        uptime=uptime,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(
        f"Unhandled exception for {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "type": type(exc).__name__,
        },
    )


def run_server(
    host: str | None = None,
    port: int | None = None,
    reload: bool | None = None,
):
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to (default: from config or 0.0.0.0)
        port: Port to listen on (default: from config or 8000)
        reload: Enable auto-reload (default: from config or False)
    """
    config = default_config

    # Use provided values or fall back to config
    final_host = host or config.server.host
    final_port = port or config.server.port
    final_reload = reload if reload is not None else config.server.reload

    logger.info("=" * 80)
    logger.info("STARTING VISUAL SEARCH API SERVER".center(80))
    logger.info("=" * 80)
    logger.info(f"Host: {final_host}")
    logger.info(f"Port: {final_port}")
    logger.info(f"Reload: {final_reload}")
    logger.info(f"Index folder: {config.data.index_dir}")
    logger.info("=" * 80)

    uvicorn.run(
        "backend.server:app",
        host=final_host,
        port=final_port,
        reload=final_reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
