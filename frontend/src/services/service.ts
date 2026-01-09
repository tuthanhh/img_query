import type { SearchResult } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Performs a visual search by calling the backend API
 * @param queryText - Text query for search (can be empty if using image)
 * @param queryImage - Base64 encoded image (without data URI prefix)
 * @param likedItems - Array of previously liked/relevant search results for refinement
 * @param dislikedItems - Array of previously disliked/irrelevant search results for refinement
 * @param positiveText - Array of positive keywords to include in search
 * @param negativeText - Array of negative keywords to exclude from search
 */
export const performSearchBackend = async (
  queryText: string,
  queryImage: string | null,
  likedItems: SearchResult[],
  dislikedItems: SearchResult[],
  positiveText: string[],
  negativeText: string[],
  k: number,
): Promise<SearchResult[]> => {
  try {
    const requestBody = {
      query: queryText,
      image: queryImage,
      liked_items: likedItems,
      disliked_items: dislikedItems,
      positive_text: positiveText,
      negative_text: negativeText,
      k: k,
    };

    const response = await fetch(`${API_BASE_URL}/api/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.message ||
          `Server error: ${response.status} with response ${response}`,
      );
    }

    const data = await response.json();
    return data.results || [];
  } catch (error) {
    console.error("Search Error:", error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(
      "Failed to perform search. Please check your connection and try again.",
    );
  }
};

const ADJECTIVES = [
  "Vintage",
  "Modern",
  "Abstract",
  "Neon",
  "Gloomy",
  "Sunny",
  "Urban",
  "Wild",
  "Techno",
  "Rustic",
  "Cyberpunk",
  "Minimalist",
  "Gothic",
  "Vibrant",
];
const NOUNS = [
  "Landscape",
  "Portrait",
  "Architecture",
  "Nature",
  "Cityscape",
  "Texture",
  "Pattern",
  "Sky",
  "Ocean",
  "Mountain",
  "Robot",
  "Interior",
  "Forest",
  "Desert",
];

export const generateRandomQuery = (): string => {
  const adj = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
  const noun = NOUNS[Math.floor(Math.random() * NOUNS.length)];
  return `${adj} ${noun}`;
};

/**
 * Performs a random search by generating a random query
 * @param k - Number of results to retrieve
 */
export const performRandomSearch = async (
  k: number,
): Promise<SearchResult[]> => {
  const randomQuery = generateRandomQuery();
  return performSearchBackend(randomQuery, null, [], [], [], [], k);
};
