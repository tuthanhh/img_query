import React, { useRef } from 'react';
import { SearchIcon, UploadIcon, XIcon, LoadingSpinner } from './Icon';

interface SearchBarProps {
  query: string;
  setQuery: (q: string) => void;
  imageBase64: string | null;
  setImageBase64: (base64: string | null) => void;
  onSearch: () => void;
  isLoading: boolean;
  className?: string;
}

const SearchBar: React.FC<SearchBarProps> = ({ 
  query, 
  setQuery, 
  imageBase64, 
  setImageBase64, 
  onSearch, 
  isLoading,
  className 
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      onSearch();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result as string;
        // Strip prefix if necessary, but Gemini mostly handles data urls or raw base64. 
        // We'll keep the data URL format (e.g. data:image/jpeg;base64,...) as standard Gemini helpers often prefer stripping it, 
        // but @google/genai inlineData prefers base64 only.
        const base64Data = base64String.split(',')[1];
        setImageBase64(base64Data);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className={`relative flex items-center w-full max-w-2xl bg-white rounded-full shadow-md border border-gray-200 transition-shadow focus-within:shadow-lg ${className}`}>
      <div className="pl-5 text-gray-400">
        <SearchIcon className="w-5 h-5" />
      </div>

      <input
        type="text"
        className="flex-grow w-full py-3 px-4 text-gray-700 bg-transparent border-none outline-none placeholder-gray-400"
        placeholder="Search images or paste image URL..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isLoading}
      />

      {imageBase64 && (
        <div className="relative mr-2 group">
          <img 
            src={`data:image/jpeg;base64,${imageBase64}`} 
            alt="Preview" 
            className="w-8 h-8 rounded object-cover border border-gray-200"
          />
          <button 
            onClick={() => {
              setImageBase64(null);
              if (fileInputRef.current) fileInputRef.current.value = '';
            }}
            className="absolute -top-1 -right-1 bg-gray-800 text-white rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <XIcon className="w-3 h-3" />
          </button>
        </div>
      )}

      <div className="flex items-center pr-3 border-l border-gray-200 pl-3 space-x-2">
        <button 
          className="p-2 text-gray-500 hover:text-primary transition-colors rounded-full hover:bg-gray-100"
          onClick={() => fileInputRef.current?.click()}
          title="Upload an image"
          disabled={isLoading}
        >
          <UploadIcon className="w-5 h-5" />
        </button>
        <input 
          type="file" 
          ref={fileInputRef} 
          className="hidden" 
          accept="image/*"
          onChange={handleFileChange}
        />
        
        <button
          onClick={onSearch}
          disabled={isLoading}
          className="bg-primary text-white p-2 rounded-full hover:bg-blue-600 transition-colors disabled:bg-blue-300"
        >
          {isLoading ? <LoadingSpinner className="w-5 h-5" /> : <SearchIcon className="w-5 h-5" />}
        </button>
      </div>
    </div>
  );
};

export default SearchBar;