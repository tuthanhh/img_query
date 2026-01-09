import React from "react";
import type { SearchResult } from "../types";
import { FeedbackType } from "../types";
import { ThumbsUpIcon, ThumbsDownIcon } from "./Icon";

interface ImageCardProps {
  result: SearchResult;
  feedback: FeedbackType;
  onFeedback: (type: FeedbackType) => void;
}

const ImageCard: React.FC<ImageCardProps> = ({
  result,
  feedback,
  onFeedback,
}) => {
  // 1. Prepend the Data URI header (adjust 'image/jpeg' or 'image/png' as needed)
  const imageSource = `data:image/jpeg;base64,${result.image_data}`;

  return (
    <div
      className={`group relative rounded-xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-300 bg-white border ${feedback === FeedbackType.LIKE ? "border-success ring-2 ring-success ring-opacity-50" : feedback === FeedbackType.DISLIKE ? "border-secondary ring-2 ring-secondary ring-opacity-50" : "border-gray-100"}`}
    >
      {/* Image Container */}
      <div className="aspect-[4/3] overflow-hidden bg-gray-100 relative">
        <img
          src={imageSource}
          alt={result.description}
          className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-500"
          loading="lazy"
        />

        {/* Overlay Gradient on Hover */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      </div>

      {/* Content */}
      <div className="p-3">
        <p className="text-sm text-gray-700 font-medium line-clamp-2 leading-snug">
          {result.description}
        </p>
        <p className="text-xs text-gray-400 mt-1 line-clamp-1">
          {result.relevanceReason}
        </p>
      </div>

      {/* Action Buttons - Always visible on mobile, visible on hover for desktop */}
      <div className="absolute top-2 right-2 flex space-x-2">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onFeedback(
              feedback === FeedbackType.LIKE
                ? FeedbackType.NEUTRAL
                : FeedbackType.LIKE,
            );
          }}
          className={`p-2 rounded-full backdrop-blur-md shadow-sm transition-colors ${
            feedback === FeedbackType.LIKE
              ? "bg-success text-white"
              : "bg-white/90 text-gray-600 hover:text-success hover:bg-white"
          }`}
          title="More like this"
        >
          <ThumbsUpIcon
            className="w-4 h-4"
            filled={feedback === FeedbackType.LIKE}
          />
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onFeedback(
              feedback === FeedbackType.DISLIKE
                ? FeedbackType.NEUTRAL
                : FeedbackType.DISLIKE,
            );
          }}
          className={`p-2 rounded-full backdrop-blur-md shadow-sm transition-colors ${
            feedback === FeedbackType.DISLIKE
              ? "bg-secondary text-white"
              : "bg-white/90 text-gray-600 hover:text-secondary hover:bg-white"
          }`}
          title="Less like this"
        >
          <ThumbsDownIcon
            className="w-4 h-4"
            filled={feedback === FeedbackType.DISLIKE}
          />
        </button>
      </div>
    </div>
  );
};

export default ImageCard;
