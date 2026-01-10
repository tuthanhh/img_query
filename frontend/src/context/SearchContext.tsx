import React, { createContext, useContext, useState, useCallback } from "react";
import type { ReactNode } from "react";
import type {
  SearchContextType,
  SearchState,
  FeedbackType,
  AlgorithmType,
} from "../types";
import { performSearchBackend, generateRandomQuery } from "../services/service";
const defaultState: SearchState = {
  view: "home",
  queryText: "",
  queryImage: null,
  queryImagePreview: null,
  positiveContext: "",
  negativeContext: "",
  k: 12,
  algorithmType: "standard",
  results: [],
  relevantIds: new Set(),
  irrelevantIds: new Set(),
  relevantImages: [],
  irrelevantImages: [],
  isLoading: false,
};

const SearchContext = createContext<SearchContextType | undefined>(undefined);

export const SearchProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [state, setState] = useState<SearchState>(defaultState);

  const setQueryText = (text: string) => {
    setState((prev) => ({
      ...prev,
      queryText: text,
      // If text is entered, ensure image is cleared to enforce exclusivity
      queryImage: text ? null : prev.queryImage,
      queryImagePreview: text ? null : prev.queryImagePreview,
    }));
  };

  const setQueryImage = (file: File | null) => {
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setState((prev) => ({
          ...prev,
          queryImage: file,
          queryImagePreview: reader.result as string,
          queryText: "", // Clear text if image is selected to enforce exclusivity
        }));
      };
      reader.readAsDataURL(file);
    } else {
      setState((prev) => ({
        ...prev,
        queryImage: null,
        queryImagePreview: null,
      }));
    }
  };

  const setPositiveContext = (text: string) =>
    setState((prev) => ({ ...prev, positiveContext: text }));
  const setNegativeContext = (text: string) =>
    setState((prev) => ({ ...prev, negativeContext: text }));
  const setK = (k: number) => setState((prev) => ({ ...prev, k }));
  const setAlgorithmType = (algo: AlgorithmType) =>
    setState((prev) => ({ ...prev, algorithmType: algo }));

  const toggleRelevance = (id: string, type: FeedbackType) => {
    setState((prev) => {
      const newRelevant = new Set(prev.relevantIds);
      const newIrrelevant = new Set(prev.irrelevantIds);
      let newRelevantImages = [...prev.relevantImages];
      let newIrrelevantImages = [...prev.irrelevantImages];

      // Find the image in current results
      const image = prev.results.find((r) => r.id === id);

      // Clean slate for this ID first
      newRelevant.delete(id);
      newIrrelevant.delete(id);
      newRelevantImages = newRelevantImages.filter((img) => img.id !== id);
      newIrrelevantImages = newIrrelevantImages.filter((img) => img.id !== id);

      if (type && type.type === "relevant" && image) {
        newRelevant.add(id);
        newRelevantImages.push(image);
      } else if (type && type.type === "irrelevant" && image) {
        newIrrelevant.add(id);
        newIrrelevantImages.push(image);
      }

      return {
        ...prev,
        relevantIds: newRelevant,
        irrelevantIds: newIrrelevant,
        relevantImages: newRelevantImages,
        irrelevantImages: newIrrelevantImages,
      };
    });
  };

  const removeFeedback = (id: string) => {
    toggleRelevance(id, { type: null });
  };

  const performSearch = useCallback(async () => {
    setState((prev) => ({ ...prev, isLoading: true }));
    try {
      // Convert File to base64 if exists
      let imageBase64: string | null = null;
      if (state.queryImage) {
        const reader = new FileReader();
        imageBase64 = await new Promise<string>((resolve) => {
          reader.onloadend = () => {
            const base64String = reader.result as string;
            const base64Data = base64String.split(",")[1];
            resolve(base64Data);
          };
          reader.readAsDataURL(state.queryImage as Blob);
        });
      }

      // Split positive and negative context into arrays
      const positiveText = state.positiveContext
        ? state.positiveContext
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean)
        : [];
      const negativeText = state.negativeContext
        ? state.negativeContext
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean)
        : [];

      const results = await performSearchBackend(
        state.queryText,
        imageBase64,
        [], // No liked items in initial search
        [], // No disliked items in initial search
        positiveText,
        negativeText,
        state.k,
        state.algorithmType,
      );

      setState((prev) => ({
        ...prev,
        results: results,
        relevantIds: new Set(), // Reset feedback on new primary search
        irrelevantIds: new Set(),
        relevantImages: [],
        irrelevantImages: [],
        view: "results",
        isLoading: false,
      }));
    } catch (error) {
      console.error("Search failed:", error);
      setState((prev) => ({
        ...prev,
        isLoading: false,
      }));
    }
  }, [
    state.queryText,
    state.queryImage,
    state.positiveContext,
    state.negativeContext,
    state.k,
    state.algorithmType,
  ]);

  const performRandomSearch = useCallback(async () => {
    const randomQuery = generateRandomQuery();

    // Update state with the random query and start loading immediately
    setState((prev) => ({
      ...prev,
      queryText: randomQuery,
      queryImage: null,
      queryImagePreview: null,
      isLoading: true,
      view: "results",
    }));

    try {
      // Split positive and negative context into arrays
      const positiveText = state.positiveContext
        ? state.positiveContext
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean)
        : [];
      const negativeText = state.negativeContext
        ? state.negativeContext
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean)
        : [];
      const results = await performSearchBackend(
        randomQuery,
        null,
        [],
        [],
        positiveText,
        negativeText,
        state.k,
        state.algorithmType,
      );

      setState((prev) => ({
        ...prev,
        results: results,
        relevantIds: new Set(),
        irrelevantIds: new Set(),
        relevantImages: [],
        irrelevantImages: [],
        isLoading: false,
      }));
    } catch (error) {
      console.error("Random search failed:", error);
      setState((prev) => ({
        ...prev,
        isLoading: false,
      }));
    }
  }, [
    state.positiveContext,
    state.negativeContext,
    state.k,
    state.algorithmType,
  ]);

  const refineSearch = useCallback(async () => {
    setState((prev) => ({ ...prev, isLoading: true }));
    try {
      // Convert File to base64 if exists
      let imageBase64: string | null = null;
      if (state.queryImage !== null) {
        const reader = new FileReader();
        imageBase64 = await new Promise<string>((resolve) => {
          reader.onloadend = () => {
            const base64String = reader.result as string;
            const base64Data = base64String.split(",")[1];
            resolve(base64Data);
          };
          reader.readAsDataURL(state.queryImage!);
        });
      }

      // Split positive and negative context into arrays
      const positiveText = state.positiveContext
        ? state.positiveContext
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean)
        : [];
      const negativeText = state.negativeContext
        ? state.negativeContext
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean)
        : [];

      // Get relevant and irrelevant items from current results
      const likedItems = state.results.filter((r) =>
        state.relevantIds.has(r.id),
      );
      const dislikedItems = state.results.filter((r) =>
        state.irrelevantIds.has(r.id),
      );

      const results = await performSearchBackend(
        state.queryText,
        imageBase64,
        likedItems,
        dislikedItems,
        positiveText,
        negativeText,
        state.k,
        state.algorithmType,
      );

      setState((prev) => ({
        ...prev,
        results: results,
        isLoading: false,
      }));
    } catch (error) {
      console.error("Refine search failed:", error);
      setState((prev) => ({
        ...prev,
        isLoading: false,
      }));
    }
  }, [
    state.queryText,
    state.queryImage,
    state.positiveContext,
    state.negativeContext,
    state.relevantIds,
    state.irrelevantIds,
    state.results,
    state.k,
    state.algorithmType,
  ]);

  const resetSearch = () => {
    setState(defaultState);
  };

  return (
    <SearchContext.Provider
      value={{
        ...state,
        setQueryText,
        setQueryImage,
        setPositiveContext,
        setNegativeContext,
        setK,
        setAlgorithmType,
        toggleRelevance,
        performSearch,
        performRandomSearch,
        refineSearch,
        resetSearch,
        removeFeedback,
      }}
    >
      {children}
    </SearchContext.Provider>
  );
};

export const useSearch = () => {
  const context = useContext(SearchContext);
  if (!context) {
    throw new Error("useSearch must be used within a SearchProvider");
  }
  return context;
};
