export interface SearchResult {
  id: string;
  description: string;
  seed: string; // Used to generate the consistent placeholder image
  relevanceReason: string;
  score: number;
  image_path: string;
  image_data: string;
}

export interface FeedbackType {
  type: "relevant" | "irrelevant" | null;
}

export type AlgorithmType = "standard" | "ide_regular" | "ide_dec_hi";

export interface SearchState {
  view: "home" | "results";
  queryText: string;
  queryImage: File | null;
  queryImagePreview: string | null;
  positiveContext: string;
  negativeContext: string;
  k: number;
  results: SearchResult[];
  algorithmType: AlgorithmType;
  relevantIds: Set<string>;
  irrelevantIds: Set<string>;
  relevantImages: SearchResult[];
  irrelevantImages: SearchResult[];
  isLoading: boolean;
}

export interface SearchContextType extends SearchState {
  setQueryText: (text: string) => void;
  setQueryImage: (file: File | null) => void;
  setPositiveContext: (text: string) => void;
  setNegativeContext: (text: string) => void;
  setK: (k: number) => void;
  setAlgorithmType: (type: AlgorithmType) => void;
  toggleRelevance: (id: string, type: FeedbackType) => void;
  performSearch: () => void;
  performRandomSearch: () => void;
  refineSearch: () => void;
  resetSearch: () => void;
  removeFeedback: (id: string) => void;
}
