export interface SearchResult {
  id: string;
  description: string;
  seed: string; // Used to generate the consistent placeholder image
  relevanceReason: string;
  score: number;
  image_path: string;
  image_data: string;
}

export interface SearchState {
  query: string;
  imageBase64: string | null;
  results: SearchResult[];
  likedIds: Set<string>;
  dislikedIds: Set<string>;
  isLoading: boolean;
  hasSearched: boolean;
}

export const FeedbackType = {
  LIKE: "LIKE",
  DISLIKE: "DISLIKE",
  NEUTRAL: "NEUTRAL",
} as const;

export type FeedbackType = (typeof FeedbackType)[keyof typeof FeedbackType];
