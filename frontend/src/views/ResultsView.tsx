import React, { useState, useEffect } from "react";
import { SearchBar } from "../components/SearchBar";
import { RefinementPanel } from "../components/RefinementPanel";
import { ImageCard } from "../components/ImageCard";
import { useSearch } from "../context/SearchContext";
import { Search, SlidersHorizontal, Loader2 } from "lucide-react";

export const ResultsView: React.FC = () => {
  const { results, performSearch, resetSearch, isLoading } = useSearch();
  const [isMobilePanelOpen, setIsMobilePanelOpen] = useState(false);

  // Scroll to top on mount
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header - Sticky & Blurred */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200 shadow-sm transition-all">
        <div className="max-w-[1920px] mx-auto px-4 h-16 flex items-center justify-between gap-4">
          {/* Brand */}
          <div
            onClick={resetSearch}
            className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity flex-shrink-0"
          >
            <div className="p-1.5 bg-brand-600 rounded-lg">
              <Search size={20} className="text-white" />
            </div>
            <span className="text-xl font-bold tracking-tight text-gray-900 hidden sm:block">
              Rocchio
            </span>
          </div>

          {/* Search Bar */}
          <div className="flex-1 max-w-2xl">
            <SearchBar compact onSearch={performSearch} />
          </div>

          {/* Mobile Filter Toggle */}
          <button
            className="md:hidden p-2 text-gray-600 hover:bg-gray-100 rounded-full"
            onClick={() => setIsMobilePanelOpen(!isMobilePanelOpen)}
          >
            <SlidersHorizontal size={24} />
          </button>

          {/* User Placeholder */}
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-brand-500 to-purple-500 flex-shrink-0 hidden sm:block"></div>
        </div>
      </header>

      {/* Main Content Layout - Standard Flow with Sticky Sidebar */}
      <div className="flex-1 max-w-[1920px] mx-auto w-full p-4 md:p-6 lg:p-8 flex flex-col md:flex-row gap-6 relative">
        {/* Sidebar (Desktop) - Sticky & Floating */}
        <aside className="hidden md:block md:w-80 lg:w-96 flex-shrink-0 h-[calc(100vh-8rem)] sticky top-24 z-30">
          <RefinementPanel />
        </aside>

        {/* Sidebar (Mobile Overlay) */}
        {isMobilePanelOpen && (
          <div className="fixed inset-0 z-50 md:hidden flex justify-end">
            <div
              className="fixed inset-0 bg-black/50 backdrop-blur-sm"
              onClick={() => setIsMobilePanelOpen(false)}
            ></div>
            <div className="relative w-4/5 max-w-xs bg-white h-full shadow-2xl animate-in slide-in-from-right duration-300">
              <div className="h-full">
                <RefinementPanel mobile />
              </div>
              <button
                onClick={() => setIsMobilePanelOpen(false)}
                className="absolute top-2 right-2 p-2 bg-white rounded-full shadow-md text-gray-500 z-50"
              >
                <SlidersHorizontal size={20} />
              </button>
            </div>
          </div>
        )}

        {/* Results Grid */}
        <main className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-2xl font-bold text-gray-800">
                Search Results
              </h1>
              <p className="text-sm text-gray-500 mt-1">
                Found {results.length} images matching your criteria
              </p>
            </div>

            {isLoading && (
              <div className="flex items-center gap-2 text-brand-600 font-medium animate-pulse">
                <Loader2 size={20} className="animate-spin" />
                <span className="text-sm">Processing...</span>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {results.map((result) => (
              <ImageCard key={result.id} image={result} />
            ))}
          </div>

          {results.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center py-20 text-gray-400 bg-white rounded-2xl border border-dashed border-gray-200 mt-4">
              <Search size={64} className="mb-4 opacity-20" />
              <p className="text-lg">
                No results found. Try refining your query.
              </p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};
