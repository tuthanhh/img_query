import React, { useState } from "react";
import SearchBar from "./components/SearchBar";
import ImageCard from "./components/ImageCard";
import { performSearch } from "./services/service";
import type { SearchResult } from "./types";
import { FeedbackType } from "./types";

// Using constants for initial state
const INITIAL_STATE = {
  query: "",
  imageBase64: null,
  results: [] as SearchResult[],
  isLoading: false,
  hasSearched: false,
};

const App: React.FC = () => {
  const [query, setQuery] = useState(INITIAL_STATE.query);
  const [imageBase64, setImageBase64] = useState<string | null>(
    INITIAL_STATE.imageBase64,
  );
  const [results, setResults] = useState<SearchResult[]>(INITIAL_STATE.results);
  const [isLoading, setIsLoading] = useState(INITIAL_STATE.isLoading);
  const [hasSearched, setHasSearched] = useState(INITIAL_STATE.hasSearched);
  const [error, setError] = useState<string | null>(null);

  // Track feedback: Map of resultId -> FeedbackType
  const [feedbackMap, setFeedbackMap] = useState<Record<string, FeedbackType>>(
    {},
  );

  const handleSearch = async (isRefinement = false) => {
    if (!query.trim() && !imageBase64) return;

    setIsLoading(true);
    setError(null);
    try {
      // If refinement, filter liked/disliked items from CURRENT results
      const likedItems = isRefinement
        ? results.filter((r) => feedbackMap[r.id] === FeedbackType.LIKE)
        : [];
      const dislikedItems = isRefinement
        ? results.filter((r) => feedbackMap[r.id] === FeedbackType.DISLIKE)
        : [];

      const newResults = await performSearch(
        query,
        imageBase64,
        likedItems,
        dislikedItems,
      );

      setResults(newResults);
      setHasSearched(true);
      // Reset feedback map for new results
      setFeedbackMap({});

      // Scroll to top
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = (resultId: string, type: FeedbackType) => {
    setFeedbackMap((prev) => ({
      ...prev,
      [resultId]: type,
    }));
  };

  const activeFeedbackCount = Object.values(feedbackMap).filter(
    (v) => v !== FeedbackType.NEUTRAL,
  ).length;

  return (
    <div className="min-h-screen flex flex-col bg-[#F8F9FA] text-slate-900">
      {/* Header / Sticky Search */}
      <header
        className={`sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200 transition-all duration-300 ${hasSearched ? "py-3" : "py-0 border-none"}`}
      >
        <div
          className={`container mx-auto px-4 ${hasSearched ? "flex items-center justify-between gap-4" : "hidden"}`}
        >
          <div
            className="flex items-center gap-2 cursor-pointer"
            onClick={() => {
              setHasSearched(false);
              setResults([]);
              setQuery("");
              setImageBase64(null);
              setFeedbackMap({});
            }}
          >
            <div className="w-8 h-8 bg-gradient-to-tr from-primary to-blue-400 rounded-lg flex items-center justify-center text-white font-bold text-lg">
              V
            </div>
            <span className="font-semibold text-xl hidden sm:block tracking-tight text-gray-700">
              VisualFinder
            </span>
          </div>
          <SearchBar
            query={query}
            setQuery={setQuery}
            imageBase64={imageBase64}
            setImageBase64={setImageBase64}
            onSearch={() => handleSearch(false)}
            isLoading={isLoading}
            className="flex-grow max-w-xl mx-auto shadow-sm hover:shadow-md border-gray-200"
          />
          <div className="w-8 hidden sm:block"></div> {/* Spacer for balance */}
        </div>
      </header>

      <main className="flex-grow flex flex-col">
        {/* Hero Section (Only visible when no search performed) */}
        {!hasSearched && (
          <div className="flex-grow flex flex-col items-center justify-center p-4 -mt-20">
            <div className="mb-8 flex flex-col items-center">
              <div className="w-20 h-20 bg-gradient-to-tr from-primary to-blue-400 rounded-2xl flex items-center justify-center text-white font-bold text-5xl mb-6 shadow-xl shadow-blue-200">
                V
              </div>
              <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-3 text-center tracking-tight">
                Visual Finder
              </h1>
              <p className="text-gray-500 text-lg max-w-md text-center">
                Search with text or images. Refine your results with feedback to
                find exactly what you're looking for.
              </p>
            </div>

            <SearchBar
              query={query}
              setQuery={setQuery}
              imageBase64={imageBase64}
              setImageBase64={setImageBase64}
              onSearch={() => handleSearch(false)}
              isLoading={isLoading}
              className="w-full max-w-2xl text-lg shadow-lg hover:shadow-xl py-1"
            />

            <div className="mt-8 flex gap-4 text-sm text-gray-500">
              <span>Try:</span>
              <button
                onClick={() => setQuery("Vintage cyberpunk city")}
                className="hover:text-primary underline"
              >
                Vintage cyberpunk city
              </button>
              <button
                onClick={() => setQuery("Minimalist wooden furniture")}
                className="hover:text-primary underline"
              >
                Minimalist furniture
              </button>
              <button
                onClick={() => setQuery("Abstract geometric patterns")}
                className="hover:text-primary underline"
              >
                Abstract patterns
              </button>
            </div>
          </div>
        )}

        {/* Results Section */}
        {hasSearched && (
          <div className="container mx-auto px-4 py-8 relative">
            {error && (
              <div className="bg-red-50 text-red-600 p-4 rounded-lg mb-6 border border-red-100 text-center">
                {error}
              </div>
            )}

            {isLoading ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 animate-pulse">
                {[...Array(8)].map((_, i) => (
                  <div
                    key={i}
                    className="aspect-[4/3] bg-gray-200 rounded-xl"
                  ></div>
                ))}
              </div>
            ) : results.length > 0 ? (
              <>
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold text-gray-800">
                    Top Results
                    <span className="text-sm font-normal text-gray-500 ml-2">
                      found {results.length} visual matches
                    </span>
                  </h2>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 pb-24">
                  {results.map((result) => (
                    <ImageCard
                      key={result.id}
                      result={result}
                      feedback={feedbackMap[result.id] || FeedbackType.NEUTRAL}
                      onFeedback={(type) => handleFeedback(result.id, type)}
                    />
                  ))}
                </div>
              </>
            ) : (
              <div className="text-center py-20 text-gray-500">
                <p>No results found. Try a different query.</p>
              </div>
            )}
          </div>
        )}

        {/* Floating Refine Button */}
        {hasSearched && !isLoading && results.length > 0 && (
          <div
            className={`fixed bottom-8 left-1/2 transform -translate-x-1/2 transition-all duration-500 z-40 ${activeFeedbackCount > 0 ? "translate-y-0 opacity-100" : "translate-y-20 opacity-0 pointer-events-none"}`}
          >
            <button
              onClick={() => handleSearch(true)}
              className="flex items-center gap-3 bg-gray-900 text-white px-8 py-3 rounded-full shadow-2xl hover:bg-black transition-colors ring-4 ring-white"
            >
              <span className="flex items-center gap-1.5">
                <span className="bg-success/20 text-success-300 rounded px-1.5 py-0.5 text-xs font-bold bg-gray-700">
                  {
                    Object.values(feedbackMap).filter(
                      (v) => v === FeedbackType.LIKE,
                    ).length
                  }
                </span>
                Likes
              </span>
              <span className="w-px h-4 bg-gray-600"></span>
              <span className="flex items-center gap-1.5">
                <span className="bg-secondary/20 text-secondary-300 rounded px-1.5 py-0.5 text-xs font-bold bg-gray-700">
                  {
                    Object.values(feedbackMap).filter(
                      (v) => v === FeedbackType.DISLIKE,
                    ).length
                  }
                </span>
                Dislikes
              </span>
              <span className="ml-2 font-semibold text-blue-200">
                Refine Results &rarr;
              </span>
            </button>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-100 py-6 mt-auto">
        <div className="container mx-auto px-4 text-center text-sm text-gray-400">
          <p>© {new Date().getFullYear()} Visual Finder</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
