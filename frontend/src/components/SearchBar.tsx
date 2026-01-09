import React, { useRef } from 'react';
import { Search, Image as ImageIcon, X } from 'lucide-react';
import { useSearch } from '../context/SearchContext';

interface SearchBarProps {
  compact?: boolean;
  onSearch?: () => void;
}

export const SearchBar: React.FC<SearchBarProps> = ({ compact = false, onSearch }) => {
  const { queryText, setQueryText, queryImagePreview, setQueryImage, queryImage } = useSearch();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && onSearch) {
      onSearch();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setQueryImage(e.target.files[0]);
    }
  };

  const clearImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    setQueryImage(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const hasImage = !!queryImagePreview;

  return (
    <div className={`relative flex items-center w-full transition-all duration-300 ${compact ? 'h-12' : 'h-14'} bg-white border border-gray-200 rounded-full shadow-sm hover:shadow-md focus-within:shadow-md focus-within:border-brand-300`}>
      {/* Search Icon */}
      <div className="pl-4 pr-2 text-gray-400">
        <Search size={compact ? 20 : 24} />
      </div>

      {/* Image Preview Pill */}
      {queryImagePreview && (
        <div className="flex items-center gap-2 pl-1 pr-3 py-1 mr-2 bg-gray-100 rounded-full text-xs font-medium text-gray-700 animate-in fade-in zoom-in duration-200">
          <img src={queryImagePreview} alt="Preview" className="w-6 h-6 rounded-full object-cover" />
          <span className="max-w-[80px] truncate">{queryImage?.name}</span>
          <button onClick={clearImage} className="hover:text-red-500 rounded-full p-0.5">
            <X size={14} />
          </button>
        </div>
      )}

      {/* Text Input */}
      <input
        type="text"
        className={`flex-1 bg-transparent border-none outline-none text-gray-800 placeholder-gray-400 text-base ${hasImage ? 'cursor-not-allowed opacity-50' : ''}`}
        placeholder={hasImage ? "Searching by image..." : "Describe what you are looking for..."}
        value={queryText}
        onChange={(e) => setQueryText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={hasImage}
      />

      {/* Upload Button */}
      <div className="pr-2">
        <button
          onClick={() => fileInputRef.current?.click()}
          className={`p-2 rounded-full transition-colors ${hasImage ? 'text-brand-600 bg-brand-50' : 'text-gray-400 hover:text-brand-600 hover:bg-brand-50'}`}
          title="Search by image"
        >
          <ImageIcon size={compact ? 20 : 24} />
        </button>
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept="image/*"
          onChange={handleFileChange}
        />
      </div>

      {/* Primary Search Button (Only visible in compact/header mode, usually invisible in large home mode as that has a separate button) */}
      {compact && onSearch && (
        <button
            onClick={onSearch}
            className="mr-1 px-4 py-1.5 bg-brand-600 text-white rounded-full text-sm font-medium hover:bg-brand-700 transition-colors"
        >
            Search
        </button>
      )}
    </div>
  );
};