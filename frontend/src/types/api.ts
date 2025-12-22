/**
 * API Types - Match FastAPI Pydantic models
 */

// ==================== Movie Types ====================

export interface MovieRecommendation {
  movie_id: number;
  title: string;
  score: number;
  genres: string[];
  year: number | null;
  avg_rating: number | null;
  popularity: number | null;
}

export interface MovieSearchResult {
  movie_id: number;
  title: string;
  genres: string[];
  year: number | null;
  relevance_score: number;
}

// ==================== Request Types ====================

export interface RecommendationRequest {
  user_id: number;
  k?: number;
  min_rating?: number;
  genres?: string[];
  min_year?: number;
  max_year?: number;
}

export interface ColdStartRating {
  movie_id: number;
  rating: number;
}

export interface ColdStartRequest {
  ratings: ColdStartRating[];
}

// ==================== Response Types ====================

export interface RecommendationResponse {
  user_id: number;
  recommendations: MovieRecommendation[];
  count: number;
  cached: boolean;
  filters_applied: Record<string, unknown> | null;
}

export interface ColdStartResponse {
  temp_user_id: string;
  recommendations: MovieRecommendation[];
  count: number;
  ratings_processed: number;
}

export interface SimilarMoviesResponse {
  movie_id: number;
  query_title: string;
  similar_movies: MovieRecommendation[];
  count: number;
}

export interface PopularMoviesResponse {
  popular_movies: MovieRecommendation[];
  count: number;
  genre_filter: string | null;
  cached: boolean;
}

export interface MovieSearchResponse {
  query: string;
  results: MovieSearchResult[];
  count: number;
}

// ==================== Health Types ====================

export interface ServiceHealth {
  service: string;
  healthy: boolean;
  latency_ms: number | null;
  details: Record<string, unknown> | null;
}

export interface DeepHealthResponse {
  status: string;
  services: ServiceHealth[];
  timestamp: string;
}

export interface SystemStats {
  total_users: number;
  total_movies: number;
  cache_stats: Record<string, unknown>;
  qdrant_stats: Record<string, unknown>;
  timestamp: string;
}

// ==================== Error Types ====================

export interface APIError {
  error: string;
  detail?: string;
  error_code?: string;
}
