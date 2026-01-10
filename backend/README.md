# Backend - Image Query System API

FastAPI backend providing RESTful API for visual search using CLIP embeddings and FAISS indexing.

## Installation

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Setup directories
make setup-dirs
```

## Configuration

Create `.env` file:

```env
CLIP_MODEL_NAME=ViT-B/32
DEVICE=auto
USE_GPU=true
INDEX_TYPE=flat
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
BATCH_SIZE=32
GEMINI_API_KEY=your_key_here  # Optional
```

## CLI Commands

### Server

```bash
# Start server
make server
uv run backend server

# Custom port
uv run backend server --port 8080

# Verbose logging
uv run backend server -v
```

### Data Management

```bash
# Split dataset (train/test)
make split
uv run backend split --input data/archive --ratio 0.2

# Create FAISS index
make index
uv run backend index --input data/train --output data/index

# View statistics
make stats
uv run backend stats
```

### Search (CLI)

```bash
# Search by text
uv run backend search --text "red car" --k 10

# Search by image
uv run backend search --image photo.jpg --k 10

# Save results
uv run backend search --text "sunset" --output results.json
```

## API Endpoints

Base URL: `http://localhost:8000`

### POST /api/search

Search for images using text or image query with relevance feedback.

**Request:**
```json
{
  "query": "red sports car",
  "image": null,
  "k": 12,
  "liked_items": [
    {"image_path": "data/train/img1.jpg"}
  ],
  "disliked_items": [
    {"image_path": "data/train/img2.jpg"}
  ],
  "positive_text": ["fast", "modern"],
  "negative_text": ["old", "damaged"]
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "img123",
      "description": "Image: car_red.jpg",
      "score": 0.8543,
      "image_path": "data/train/car_red.jpg",
      "image_data": "base64_encoded..."
    }
  ],
  "total": 12,
  "query": "red sports car"
}
```

**cURL Example (Text):**
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "red car", "k": 5}'
```

**cURL Example (Image):**
```bash
IMAGE_BASE64=$(base64 -w 0 image.jpg)
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_BASE64\", \"k\": 5}"
```

**cURL Example (Relevance Feedback):**
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "modern architecture",
    "k": 10,
    "liked_items": [{"image_path": "data/train/building1.jpg"}],
    "disliked_items": [{"image_path": "data/train/old_house.jpg"}],
    "positive_text": ["glass facade"],
    "negative_text": ["traditional"]
  }'
```

### GET /api/stats

Get index statistics.

**Response:**
```json
{
  "total_images": 10000,
  "index_size": 2048000,
  "embedding_dim": 512,
  "index_type": "flat",
  "model_name": "ViT-B/32",
  "created_at": "2024-01-15T10:30:45Z",
  "device": "cuda"
}
```

### GET /health

Health check.

**Response:**
```json
{
  "status": "healthy",
  "search_engine": "available",
  "uptime": 3600.5
}
```

### GET /

API information.

**Response:**
```json
{
  "message": "Visual Search API",
  "version": "0.1.0",
  "status": "ready",
  "search_available": true
}
```

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs

## Command Reference

| Command | Description |
|---------|-------------|
| `make server` | Start API server |
| `make index` | Create FAISS index |
| `make split` | Split dataset |
| `make stats` | Show index stats |
| `make format` | Format code |
| `make lint` | Run linting |
| `make test` | Run tests |
| `make clean` | Clean cache |

## Architecture

```
backend/
├── api/
│   └── routes.py       # API endpoints
├── config.py           # Configuration
├── indexer.py          # Index creation
├── search.py           # Search engine
├── server.py           # FastAPI app
├── schema.py           # Pydantic models
└── utils.py            # Helpers
```

---

**See main README for general project overview**
