import React, { useState } from "react";
import { SearchBar } from "../components/SearchBar";
import { useSearch } from "../context/SearchContext";
import { Search, ChevronDown, ChevronUp, Loader2 } from "lucide-react";

export const HomeView: React.FC = () => {
  const {
    performSearch,
    performRandomSearch,
    positiveContext,
    setPositiveContext,
    negativeContext,
    setNegativeContext,
    k,
    setK,
    isLoading,
  } = useSearch();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activeSearchType, setActiveSearchType] = useState<
    "standard" | "lucky" | null
  >(null);

  const handleStandardSearch = () => {
    setActiveSearchType("standard");
    performSearch();
  };

  const handleLuckySearch = () => {
    setActiveSearchType("lucky");
    performRandomSearch();
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-white p-4 animate-in fade-in duration-500">
      <div className="w-full max-w-2xl flex flex-col items-center">
        {/* Logo / Brand */}
        <div className="flex items-center gap-3 mb-10">
          <div className="p-3 bg-brand-600 rounded-xl shadow-lg shadow-brand-500/30">
            <Search size={40} className="text-white" />
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-gray-900">
            Rocchio<span className="text-brand-600">Search</span>
          </h1>
        </div>

        {/* Main Search */}
        <div className="w-full mb-8">
          <SearchBar onSearch={handleStandardSearch} />
        </div>

        {/* Action Button */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={handleStandardSearch}
            disabled={isLoading}
            className={`min-w-[160px] flex items-center justify-center gap-2 px-8 py-3 bg-brand-600 hover:bg-brand-700 text-white font-medium rounded-md shadow-sm transition-colors ${isLoading ? "opacity-80 cursor-not-allowed" : ""}`}
          >
            {isLoading && activeSearchType === "standard" ? (
              <>
                <Loader2 size={20} className="animate-spin" />
                <span>Searching...</span>
              </>
            ) : (
              <span>Search Images</span>
            )}
          </button>
          <button
            onClick={handleLuckySearch}
            disabled={isLoading}
            className={`min-w-[160px] flex items-center justify-center gap-2 px-8 py-3 bg-gray-50 hover:bg-gray-100 text-gray-700 font-medium rounded-md border border-gray-200 transition-colors ${isLoading ? "opacity-80 cursor-not-allowed" : ""}`}
          >
            {isLoading && activeSearchType === "lucky" ? (
              <>
                <Loader2 size={20} className="animate-spin text-brand-600" />
                <span>Lucky...</span>
              </>
            ) : (
              <span>I'm Feeling Lucky</span>
            )}
          </button>
        </div>

        {/* Advanced Options Toggle */}
        <div className="w-full">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="mx-auto flex items-center gap-1 text-sm text-gray-500 hover:text-brand-600 transition-colors"
          >
            {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            {showAdvanced ? "Hide Advanced Options" : "Show Advanced Options"}
          </button>

          {showAdvanced && (
            <div className="mt-6 p-6 bg-gray-50 rounded-xl border border-gray-100 w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-in slide-in-from-top-2 duration-300 shadow-sm">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Positive Context (Boost)
                </label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-brand-500 focus:border-brand-500 text-sm transition-all shadow-sm"
                  placeholder="Keywords to include..."
                  value={positiveContext}
                  onChange={(e) => setPositiveContext(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Negative Context (Suppress)
                </label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-brand-500 focus:border-brand-500 text-sm transition-all shadow-sm"
                  placeholder="Keywords to exclude..."
                  value={negativeContext}
                  onChange={(e) => setNegativeContext(e.target.value)}
                />
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="block text-sm font-medium text-gray-700">
                    Results Count (K)
                  </label>
                  <span className="bg-brand-50 text-brand-700 border border-brand-100 px-2 py-0.5 rounded text-xs font-mono font-medium shadow-sm">
                    {k}
                  </span>
                </div>
                <div className="relative pt-1">
                  <input
                    type="range"
                    min="4"
                    max="100"
                    step="4"
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brand-600 focus:outline-none focus:ring-2 focus:ring-brand-500/20"
                    value={k}
                    onChange={(e) => setK(Number(e.target.value))}
                  />
                  <div className="flex justify-between text-[10px] text-gray-400 mt-1 font-medium">
                    <span>4</span>
                    <span>100</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="fixed bottom-4 text-xs text-gray-400">
        Rocchio v2.0 • React • Tailwind
      </div>
    </div>
  );
};
