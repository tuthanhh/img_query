# Frontend - Image Query System

React frontend for the Image Query System with text/image search and relevance feedback.

## Installation

```bash
npm install
```

## Development

```bash
# Start dev server
npm run dev              # Runs on http://localhost:5173

# Build for production
npm run build

# Preview production build
npm run preview

# Run linting
npm run lint
```

## Configuration

Update API endpoint in `src/services/api.ts`:

```typescript
const API_BASE_URL = 'http://localhost:8000';
```

## Project Structure

```
frontend/
├── src/
│   ├── components/        # UI components
│   │   ├── SearchBar.tsx
│   │   ├── ImageCard.tsx
│   │   └── RefinementPanel.tsx
│   ├── views/            # Page views
│   │   ├── HomeView.tsx
│   │   └── ResultsView.tsx
│   ├── context/          # State management
│   │   └── SearchContext.tsx
│   ├── services/         # API client
│   │   └── api.ts
│   ├── App.tsx
│   └── types.ts
├── public/
└── package.json
```

## Key Components

### SearchBar
Text and image search input component.

### ImageCard
Display individual search results with like/dislike buttons.

### RefinementPanel
Interface for refining search with relevance feedback.

## API Integration

### Search Images

```typescript
import { searchImages } from './services/api';

const results = await searchImages({
  query: "red car",
  k: 12,
  liked_items: [],
  disliked_items: [],
  positive_text: [],
  negative_text: []
});
```

### Get Stats

```typescript
import { getStats } from './services/api';

const stats = await getStats();
```

## Environment Variables

Create `.env` file:

```env
VITE_API_URL=http://localhost:8000
```

Access in code:

```typescript
const apiUrl = import.meta.env.VITE_API_URL;
```

## Deployment

### Build

```bash
npm run build
# Output in dist/
```
## Tech Stack

- React 19
- TypeScript
- Vite
- Tailwind CSS
- Lucide React (icons)

---

**See main README for project overview and backend README for API details**
