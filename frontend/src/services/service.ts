import type { SearchResult } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Performs a visual search by calling the backend API
 */
export const performSearch = async (
  query: string,
  imageBase64: string | null,
  likedItems: SearchResult[],
  dislikedItems: SearchResult[],
): Promise<SearchResult[]> => {
  try {
    const requestBody = {
      query: query,
      image: imageBase64,
      liked_items: likedItems,
      disliked_items: dislikedItems,
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
