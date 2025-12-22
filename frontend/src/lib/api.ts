/**
 * API Client for Movie Recommendation System
 */

import type {
  ColdStartRequest,
  ColdStartResponse,
  MovieSearchResponse,
  PopularMoviesResponse,
  RecommendationResponse,
  SimilarMoviesResponse,
  APIError,
} from "@/types/api";

// API base URL - defaults to localhost for development
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Custom error class for API errors
 */
export class ApiError extends Error {
  status: number;
  code?: string;
  detail?: string;

  constructor(message: string, status: number, code?: string, detail?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    let errorData: APIError;
    try {
      errorData = await response.json();
    } catch {
      throw new ApiError(
        `Request failed with status ${response.status}`,
        response.status
      );
    }
    throw new ApiError(
      errorData.error || "Unknown error",
      response.status,
      errorData.error_code,
      errorData.detail
    );
  }

  return response.json();
}

// ==================== Recommendation APIs ====================

/**
 * Get personalized recommendations for a user
 */
export async function getRecommendations(
  userId: number,
  options: {
    k?: number;
    minRating?: number;
    genres?: string[];
    minYear?: number;
    maxYear?: number;
  } = {}
): Promise<RecommendationResponse> {
  const params = new URLSearchParams();
  if (options.k) params.set("k", String(options.k));
  if (options.minRating) params.set("min_rating", String(options.minRating));
  if (options.genres?.length) params.set("genres", options.genres.join(","));
  if (options.minYear) params.set("min_year", String(options.minYear));
  if (options.maxYear) params.set("max_year", String(options.maxYear));

  const query = params.toString();
  return fetchAPI<RecommendationResponse>(
    `/api/v1/recommend/${userId}${query ? `?${query}` : ""}`
  );
}

/**
 * Get cold start recommendations based on initial ratings
 */
export async function getColdStartRecommendations(
  ratings: Array<{ movieId: number; rating: number }>
): Promise<ColdStartResponse> {
  const request: ColdStartRequest = {
    ratings: ratings.map((r) => ({
      movie_id: r.movieId,
      rating: r.rating,
    })),
  };

  return fetchAPI<ColdStartResponse>("/api/v1/cold-start/rate", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

/**
 * Get similar movies based on a movie ID
 */
export async function getSimilarMovies(
  movieId: number,
  k: number = 10
): Promise<SimilarMoviesResponse> {
  return fetchAPI<SimilarMoviesResponse>(
    `/api/v1/similar/${movieId}?k=${k}`
  );
}

// ==================== Movie APIs ====================

/**
 * Get popular movies
 */
export async function getPopularMovies(options: {
  limit?: number;
  genre?: string;
} = {}): Promise<PopularMoviesResponse> {
  const params = new URLSearchParams();
  if (options.limit) params.set("limit", String(options.limit));
  if (options.genre) params.set("genre", options.genre);

  const query = params.toString();
  return fetchAPI<PopularMoviesResponse>(
    `/api/v1/movies/popular${query ? `?${query}` : ""}`
  );
}

/**
 * Search movies by title
 */
export async function searchMovies(
  query: string,
  limit: number = 20
): Promise<MovieSearchResponse> {
  const params = new URLSearchParams({
    query,
    limit: String(limit),
  });
  return fetchAPI<MovieSearchResponse>(`/api/v1/movies/search?${params}`);
}

// ==================== Health APIs ====================

/**
 * Check API health
 */
export async function checkHealth(): Promise<{ status: string }> {
  return fetchAPI<{ status: string }>("/health");
}
