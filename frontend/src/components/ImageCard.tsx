import React from "react";
import { ThumbsUp, ThumbsDown, Check } from "lucide-react";
import type { SearchResult } from "../types";
import { useSearch } from "../context/SearchContext";

interface ImageCardProps {
  image: SearchResult;
}

export const ImageCard: React.FC<ImageCardProps> = ({ image }) => {
  const { relevantIds, irrelevantIds, toggleRelevance } = useSearch();

  const isRelevant = relevantIds.has(image.id);
  const isIrrelevant = irrelevantIds.has(image.id);
  const imageSource = `data:image/jpeg;base64,${image.image_data}`;

  let borderClass = "border-transparent";
  let opacityClass = "opacity-100";
  let shadowClass = "shadow-sm hover:shadow-xl";

  if (isRelevant) {
    borderClass = "border-green-500 ring-4 ring-green-500/20";
    shadowClass = "shadow-lg shadow-green-500/10";
  } else if (isIrrelevant) {
    borderClass = "border-red-500";
    opacityClass = "opacity-60 grayscale-[0.5]";
    shadowClass = "shadow-none";
  }

  return (
    <div
      className={`group relative rounded-xl overflow-hidden bg-white border-2 transition-all duration-300 ${borderClass} ${opacityClass} ${shadowClass}`}
    >
      {/* Image */}
      <img
        src={imageSource}
        alt={image.description}
        className="w-full h-64 object-cover"
        loading="lazy"
      />

      {/* Overlay Interactions */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-4">
        <p className="text-white font-medium text-sm mb-1 line-clamp-2">
          {image.description}
        </p>
        <p className="text-white/70 text-xs mb-3 line-clamp-1">
          {image.relevanceReason}
        </p>

        <div className="flex gap-2">
          <button
            onClick={() =>
              toggleRelevance(
                image.id,
                isRelevant ? { type: null } : { type: "relevant" },
              )
            }
            className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-semibold transition-colors backdrop-blur-md ${
              isRelevant
                ? "bg-green-500 text-white"
                : "bg-white/20 text-white hover:bg-green-500"
            }`}
          >
            {isRelevant ? <Check size={16} /> : <ThumbsUp size={16} />}
            Relevant
          </button>

          <button
            onClick={() =>
              toggleRelevance(
                image.id,
                isIrrelevant ? { type: null } : { type: "irrelevant" },
              )
            }
            className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-semibold transition-colors backdrop-blur-md ${
              isIrrelevant
                ? "bg-red-500 text-white"
                : "bg-white/20 text-white hover:bg-red-500"
            }`}
          >
            <ThumbsDown size={16} />
            Irrelevant
          </button>
        </div>
      </div>

      {/* Sticky Status Indicator (Visible when not hovering if selected) */}
      {isRelevant && (
        <div className="absolute top-3 left-3 bg-green-500 text-white p-1.5 rounded-full shadow-md z-10 group-hover:opacity-0 transition-opacity">
          <ThumbsUp size={14} fill="currentColor" />
        </div>
      )}
      {isIrrelevant && (
        <div className="absolute top-3 left-3 bg-red-500 text-white p-1.5 rounded-full shadow-md z-10 group-hover:opacity-0 transition-opacity">
          <ThumbsDown size={14} fill="currentColor" />
        </div>
      )}
    </div>
  );
};

export default ImageCard;
