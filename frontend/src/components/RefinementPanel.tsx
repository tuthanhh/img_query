import React from "react";
import { X, RefreshCw, Plus, Minus, Settings2, Sliders } from "lucide-react";
import { useSearch } from "../context/SearchContext";

interface RefinementPanelProps {
  mobile?: boolean;
}

export const RefinementPanel: React.FC<RefinementPanelProps> = ({
  mobile = false,
}) => {
  const {
    results,
    relevantIds,
    irrelevantIds,
    positiveContext,
    setPositiveContext,
    negativeContext,
    setNegativeContext,
    removeFeedback,
    refineSearch,
    isLoading,
    k,
    setK,
  } = useSearch();

  const relevantImages = results.filter((img) => relevantIds.has(img.id));
  const irrelevantImages = results.filter((img) => irrelevantIds.has(img.id));

  // Styles change based on whether it is in mobile drawer mode or desktop floating mode
  const containerClasses = mobile
    ? "h-full flex flex-col bg-white w-full"
    : "h-full flex flex-col bg-white rounded-2xl shadow-xl shadow-gray-200/50 border border-gray-200 w-full overflow-hidden transition-all hover:shadow-2xl hover:shadow-gray-200/60";

  return (
    <div className={containerClasses}>
      {/* Header */}
      <div className="p-5 border-b border-gray-100 bg-gray-50/50">
        <div className="flex items-center gap-2 text-gray-800 font-bold text-lg mb-1">
          <Settings2 size={20} className="text-brand-600" />
          <h2>Refine Results</h2>
        </div>
        <p className="text-xs text-gray-500">
          Adjust relevance parameters to train the search model (Rocchio
          Algorithm simulation).
        </p>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar p-5 space-y-8">
        {/* Retrieval Settings */}
        <div className="space-y-4">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
            <Sliders size={12} /> Configuration
          </h3>
          <div className="space-y-2">
            <div className="flex justify-between items-center text-sm">
              <label className="font-medium text-gray-700">
                Results to Retrieve (K)
              </label>
              <span className="bg-gray-100 px-2 py-0.5 rounded text-xs text-gray-600 font-mono border border-gray-200 shadow-sm">
                {k}
              </span>
            </div>
            <input
              type="range"
              min="4"
              max="100"
              step="4"
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brand-600 focus:outline-none focus:ring-2 focus:ring-brand-500/20"
            />
            <div className="flex justify-between text-xs text-gray-400 font-medium">
              <span>4</span>
              <span>100</span>
            </div>
          </div>
        </div>

        <hr className="border-gray-100" />

        {/* Text Refinement */}
        <div className="space-y-4">
          <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider">
            Semantic Context
          </h3>

          <div className="space-y-1">
            <label className="text-sm font-medium text-gray-700 flex items-center gap-1">
              <Plus size={14} className="text-relevance-600" /> Positive Context
            </label>
            <input
              type="text"
              placeholder="e.g. blue sky, ocean"
              className="w-full px-3 py-2 bg-gray-50 border border-gray-200 rounded-md text-sm focus:ring-2 focus:ring-relevance-500/20 focus:border-relevance-500 outline-none transition-all shadow-sm"
              value={positiveContext}
              onChange={(e) => setPositiveContext(e.target.value)}
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium text-gray-700 flex items-center gap-1">
              <Minus size={14} className="text-irrelevance-600" /> Negative
              Context
            </label>
            <input
              type="text"
              placeholder="e.g. cloudy, night"
              className="w-full px-3 py-2 bg-gray-50 border border-gray-200 rounded-md text-sm focus:ring-2 focus:ring-irrelevance-500/20 focus:border-irrelevance-500 outline-none transition-all shadow-sm"
              value={negativeContext}
              onChange={(e) => setNegativeContext(e.target.value)}
            />
          </div>
        </div>

        {/* Relevant Images List */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider">
              Relevant Samples ({relevantIds.size})
            </h3>
          </div>

          {relevantImages.length === 0 ? (
            <div className="text-sm text-gray-400 italic py-4 border-2 border-dashed border-gray-100 rounded-lg text-center bg-gray-50/50">
              No images marked relevant yet.
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-2">
              {relevantImages.map((img) => (
                <div key={img.id} className="relative group aspect-square">
                  <img
                    src={`data:image/jpeg;base64,${img.image_data}`}
                    alt="Relevant"
                    className="w-full h-full object-cover rounded-md border border-relevance-200 shadow-sm"
                  />
                  <button
                    onClick={() => removeFeedback(img.id)}
                    className="absolute -top-1.5 -right-1.5 bg-white text-gray-500 hover:text-red-500 rounded-full shadow-md p-0.5 opacity-0 group-hover:opacity-100 transition-all scale-90 group-hover:scale-100 border border-gray-100"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Irrelevant Images List */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider">
              Irrelevant Samples ({irrelevantIds.size})
            </h3>
          </div>

          {irrelevantImages.length === 0 ? (
            <div className="text-sm text-gray-400 italic py-4 border-2 border-dashed border-gray-100 rounded-lg text-center bg-gray-50/50">
              No images marked irrelevant yet.
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-2">
              {irrelevantImages.map((img) => (
                <div key={img.id} className="relative group aspect-square">
                  <img
                    src={`data:image/jpeg;base64,${img.image_data}`}
                    alt="Irrelevant"
                    className="w-full h-full object-cover rounded-md border border-irrelevance-200 opacity-70 shadow-sm"
                  />
                  <button
                    onClick={() => removeFeedback(img.id)}
                    className="absolute -top-1.5 -right-1.5 bg-white text-gray-500 hover:text-red-500 rounded-full shadow-md p-0.5 opacity-0 group-hover:opacity-100 transition-all scale-90 group-hover:scale-100 border border-gray-100"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Footer Action */}
      <div className="p-5 border-t border-gray-200 bg-gray-50/50 backdrop-blur-sm">
        <button
          onClick={refineSearch}
          disabled={isLoading}
          className={`w-full flex items-center justify-center gap-2 py-3 rounded-lg text-white font-semibold transition-all shadow-lg shadow-brand-500/30 ${
            isLoading
              ? "bg-brand-400 cursor-not-allowed"
              : "bg-brand-600 hover:bg-brand-700 active:scale-[0.98]"
          }`}
        >
          <RefreshCw size={18} className={isLoading ? "animate-spin" : ""} />
          {isLoading ? "Recalculating..." : "Refine Search"}
        </button>
      </div>
    </div>
  );
};
