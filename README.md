# Image Query System

A visual search application for finding images using text descriptions or similar images. Built with CLIP embeddings, FAISS vector search, and a React frontend.

## Features

- **Multi-modal Search**: Text or image-based queries
- **CLIP Embeddings**: OpenAI's CLIP for image-text understanding
- **FAISS Index**: Fast similarity search
- **Relevance Feedback**: Refine results interactively
- **Gemini AI**: Optional query enrichment

## Architecture

```
┌─────────────┐    HTTP/REST    ┌─────────────┐
│   React     │ ◄─────────────► │   FastAPI   │
│  Frontend   │                 │   Backend   │
└─────────────┘                 └──────┬──────┘
                                       │
                          ┌────────────┼────────────┐
                          │            │            │
                     ┌────▼───┐   ┌───▼────┐  ┌───▼────┐
                     │  CLIP  │   │ FAISS  │  │ Gemini │
                     └────────┘   └────────┘  └────────┘
```

**Frontend**: React 19 + TypeScript + Tailwind CSS  
**Backend**: FastAPI + Python + CLIP + FAISS

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA (optional, recommended for GPU)
- [uv](https://docs.astral.sh/uv/)

### 2. Backend Setup

```bash
cd backend
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv
uv sync                                            # Install dependencies
cp .env.example .env                               # Configure (add your settings)
```
Also remember to configure the environment variables in `backend/.env`. This is important for the backend to function correctly. The description of each variables is carefully written in backend README.

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Prepare Data & Index

```bash
cd backend
mkdir -p data/archive
# Copy your images to data/archive/

make split    # Split into train/test
make index    # Create FAISS index
```

### 5. Run Application

**Terminal 1 (Backend):**
```bash
cd backend
make server   # Runs on http://localhost:8000
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev   # Runs on http://localhost:5173
```
## Usage

### Web Interface

1. Open http://localhost:5173
2. Enter text query or upload image
3. View results and refine with relevance feedback

### CLI

```bash
cd backend

# Create index
uv run backend index

# Search by text
uv run backend search --text "red car" --k 10

# Search by image  
uv run backend search --image photo.jpg --k 10

# View stats
uv run backend stats
```

## API Documentation
See [backend/README.md](backend/README.md) for detailed API docs.

## Development

**Backend:**
```bash
cd backend
make format    # Format code
make lint      # Lint code
make test      # Run tests
```

**Frontend:**
```bash
cd frontend
npm run dev    # Dev server
npm run build  # Production build
npm run lint   # Lint code
```

## Project Structure

```
a2-img_query/
├── backend/           # FastAPI server
│   ├── src/backend/   # Core modules
│   ├── data/          # Images & index
│   └── README.md      # Backend docs
├── frontend/          # React app
│   ├── src/           # Components & views
│   └── README.md      # Frontend docs
└── README.md          # This file
```

## Documentation

- [Backend Documentation](backend/README.md) - CLI commands & API endpoints
- [Frontend Documentation](frontend/README.md) - Development & deployment

---

**For detailed instructions, see the backend and frontend README files.**
