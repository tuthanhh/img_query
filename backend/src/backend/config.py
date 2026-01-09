"""Configuration management for the image retrieval system."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for CLIP model."""

    name: str = field(default_factory=lambda: os.getenv("CLIP_MODEL_NAME", "ViT-B/32"))
    embedding_dim: int = 512
    device: Literal["cuda", "cpu", "auto"] = "auto"

    def __post_init__(self):
        """Override device from environment if set."""
        device_env = os.getenv("DEVICE")
        if device_env in ("cuda", "cpu", "auto"):
            object.__setattr__(self, "device", device_env)


@dataclass
class GeminiConfig:
    """Configuration for Gemini API."""

    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    )
    enable_enrichment: bool = field(
        default_factory=lambda: os.getenv("ENABLE_ENRICHMENT", "true").lower() == "true"
    )


@dataclass
class IndexConfig:
    """Configuration for FAISS index."""

    index_type: Literal["flat", "ivf"] = "flat"
    metric: Literal["cosine", "l2"] = "cosine"
    use_gpu: bool = field(
        default_factory=lambda: os.getenv("USE_GPU", "true").lower() == "true"
    )
    nlist: int = 100  # For IVF index

    def __post_init__(self):
        """Override index_type from environment if set."""
        index_type_env = os.getenv("INDEX_TYPE")
        if index_type_env in ("flat", "ivf"):
            object.__setattr__(self, "index_type", index_type_env)


@dataclass
class DataConfig:
    """Configuration for data paths."""

    archive_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ARCHIVE_DIR", "data/archive"))
    )
    train_dir: Path = field(
        default_factory=lambda: Path(os.getenv("TRAIN_DIR", "data/train"))
    )
    test_dir: Path = field(
        default_factory=lambda: Path(os.getenv("TEST_DIR", "data/test"))
    )
    index_dir: Path = field(
        default_factory=lambda: Path(os.getenv("INDEX_DIR", "data/index"))
    )

    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        if isinstance(self.archive_dir, str):
            self.archive_dir = Path(self.archive_dir)
        if isinstance(self.train_dir, str):
            self.train_dir = Path(self.train_dir)
        if isinstance(self.test_dir, str):
            self.test_dir = Path(self.test_dir)
        if isinstance(self.index_dir, str):
            self.index_dir = Path(self.index_dir)


@dataclass
class ProcessingConfig:
    """Configuration for image processing."""

    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "32")))
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "4")))
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
    test_split_ratio: float = 0.2
    random_seed: int = 42


@dataclass
class SearchConfig:
    """Configuration for search operations."""

    default_k: int = field(default_factory=lambda: int(os.getenv("DEFAULT_K", "10")))
    max_k: int = field(default_factory=lambda: int(os.getenv("MAX_K", "100")))
    similarity_threshold: float = 0.0  # Minimum similarity score


@dataclass
class ServerConfig:
    """Configuration for API server."""

    host: str = field(default_factory=lambda: os.getenv("SERVER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("SERVER_PORT", "8000")))
    reload: bool = field(
        default_factory=lambda: os.getenv("SERVER_RELOAD", "false").lower() == "true"
    )
    cors_origins: list[str] = field(
        default_factory=lambda: os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
        ).split(",")
    )


@dataclass
class Config:
    """Main configuration class."""

    model: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    data: DataConfig = field(default_factory=DataConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            index=IndexConfig(**config_dict.get("index", {})),
            data=DataConfig(**config_dict.get("data", {})),
            processing=ProcessingConfig(**config_dict.get("processing", {})),
            search=SearchConfig(**config_dict.get("search", {})),
            server=ServerConfig(**config_dict.get("server", {})),
            gemini=GeminiConfig(**config_dict.get("gemini", {})),
        )

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls()

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "model": {
                "name": self.model.name,
                "embedding_dim": self.model.embedding_dim,
                "device": self.model.device,
            },
            "index": {
                "index_type": self.index.index_type,
                "metric": self.index.metric,
                "use_gpu": self.index.use_gpu,
                "nlist": self.index.nlist,
            },
            "data": {
                "archive_dir": str(self.data.archive_dir),
                "train_dir": str(self.data.train_dir),
                "test_dir": str(self.data.test_dir),
                "index_dir": str(self.data.index_dir),
            },
            "processing": {
                "batch_size": self.processing.batch_size,
                "num_workers": self.processing.num_workers,
                "image_extensions": self.processing.image_extensions,
                "test_split_ratio": self.processing.test_split_ratio,
                "random_seed": self.processing.random_seed,
            },
            "search": {
                "default_k": self.search.default_k,
                "max_k": self.search.max_k,
                "similarity_threshold": self.search.similarity_threshold,
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "reload": self.server.reload,
                "cors_origins": self.server.cors_origins,
            },
            "gemini": {
                "api_key": self.gemini.api_key,
                "model": self.gemini.model,
                "enable_enrichment": self.gemini.enable_enrichment,
            },
        }

    def validate(self) -> list[str]:
        """
        Validate configuration and return list of warnings/errors.

        Returns:
            List of validation messages (empty if valid)
        """
        issues = []
        # Check if data directories exist
        if not self.data.archive_dir.exists():
            issues.append(f"Archive directory does not exist: {self.data.archive_dir}")

        if self.index.use_gpu and self.model.device == "cpu":
            issues.append("GPU index requested but CPU device specified")

        if self.processing.batch_size < 1:
            issues.append(f"Invalid batch size: {self.processing.batch_size}")

        if not 0 < self.processing.test_split_ratio < 1:
            issues.append(
                f"Invalid test split ratio: {self.processing.test_split_ratio}"
            )

        if self.search.default_k > self.search.max_k:
            issues.append(
                f"default_k ({self.search.default_k}) > max_k ({self.search.max_k})"
            )
        if self.gemini.enable_enrichment and not self.gemini.api_key:
            issues.append("Gemini enrichment is enabled but GEMINI_API_KEY is not set")

        return issues


# Default configuration instance
default_config = Config()
