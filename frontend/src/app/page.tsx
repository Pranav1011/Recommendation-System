"use client";

import { useState, useEffect, useCallback } from "react";
import { Film, Sparkles, TrendingUp, Search, ArrowRight, RefreshCw } from "lucide-react";
import { MovieCard, SearchBar, StarRating, MovieGridSkeleton } from "@/components";
import type { MovieRecommendation, MovieSearchResult } from "@/types/api";
import {
  getPopularMovies,
  searchMovies,
  getColdStartRecommendations,
  getSimilarMovies,
} from "@/lib/api";
import {
  ALL_MOVIES,
  getMoviesByGenre,
  searchMoviesByTitle,
  SEED_MOVIES_FOR_RATING,
  type MovieData,
} from "@/lib/movies-data";

// Convert MovieData to MovieRecommendation
function toMovieRecommendation(movie: MovieData, score?: number): MovieRecommendation {
  return {
    movie_id: movie.movie_id,
    title: movie.title,
    genres: movie.genres,
    year: movie.year,
    score: score ?? 0.9,
    avg_rating: movie.avg_rating,
    popularity: movie.popularity,
  };
}

type View = "home" | "coldstart" | "recommendations" | "similar" | "search";

export default function Home() {
  const [view, setView] = useState<View>("home");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Cold start state
  const [ratings, setRatings] = useState<Record<number, number>>({});
  const [recommendations, setRecommendations] = useState<MovieRecommendation[]>([]);

  // Popular movies state
  const [popularMovies, setPopularMovies] = useState<MovieRecommendation[]>([]);
  const [selectedGenre, setSelectedGenre] = useState<string | null>(null);

  // Search state
  const [searchResults, setSearchResults] = useState<MovieSearchResult[]>([]);
  const [searchQuery, setSearchQuery] = useState("");

  // Similar movies state
  const [similarMovies, setSimilarMovies] = useState<MovieRecommendation[]>([]);
  const [similarToMovie, setSimilarToMovie] = useState<string>("");

  const genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance", "Thriller", "Animation", "Adventure", "Crime"];

  // Load popular movies on mount
  useEffect(() => {
    loadPopularMovies();
  }, []);

  const loadPopularMovies = async (genre?: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await getPopularMovies({ limit: 50, genre: genre || undefined });
      setPopularMovies(response.popular_movies);
      setSelectedGenre(genre || null);
    } catch {
      // Use local movie data
      const movies = getMoviesByGenre(genre || null, 50);
      setPopularMovies(movies.map(m => toMovieRecommendation(m)));
      setSelectedGenre(genre || null);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = useCallback(async (query: string) => {
    setSearchQuery(query);
    if (!query.trim()) {
      setSearchResults([]);
      setView("home");
      return;
    }

    setLoading(true);
    setError(null);
    setView("search");

    try {
      const response = await searchMovies(query, 30);
      setSearchResults(response.results);
    } catch {
      // Use local search
      const results = searchMoviesByTitle(query, 30);
      setSearchResults(results.map(m => ({
        movie_id: m.movie_id,
        title: m.title,
        genres: m.genres,
        year: m.year,
        relevance_score: 1,
      })));
    } finally {
      setLoading(false);
    }
  }, []);

  const handleRating = (movieId: number, rating: number) => {
    setRatings(prev => ({
      ...prev,
      [movieId]: rating,
    }));
  };

  const handleGetRecommendations = async () => {
    const ratingsList = Object.entries(ratings)
      .filter(([, rating]) => rating > 0)
      .map(([movieId, rating]) => ({
        movieId: parseInt(movieId),
        rating,
      }));

    if (ratingsList.length < 3) {
      setError("Please rate at least 3 movies to get recommendations");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await getColdStartRecommendations(ratingsList);
      setRecommendations(response.recommendations);
      setView("recommendations");
    } catch {
      // Generate recommendations based on rated movies' genres
      const ratedMovieIds = new Set(ratingsList.map(r => r.movieId));
      const ratedMovies = ALL_MOVIES.filter(m => ratedMovieIds.has(m.movie_id));

      // Get preferred genres from highly rated movies
      const genreScores: Record<string, number> = {};
      ratingsList.forEach(({ movieId, rating }) => {
        const movie = ALL_MOVIES.find(m => m.movie_id === movieId);
        if (movie && rating >= 3) {
          movie.genres.forEach(g => {
            genreScores[g] = (genreScores[g] || 0) + rating;
          });
        }
      });

      // Find similar movies user hasn't rated
      const unratedMovies = ALL_MOVIES.filter(m => !ratedMovieIds.has(m.movie_id));
      const scored = unratedMovies.map(m => {
        let score = 0;
        m.genres.forEach(g => {
          score += genreScores[g] || 0;
        });
        return { movie: m, score: score / (m.genres.length || 1) };
      });

      // Sort by score and take top 20
      scored.sort((a, b) => b.score - a.score);
      const topMovies = scored.slice(0, 20);
      const maxScore = topMovies[0]?.score || 1;

      setRecommendations(topMovies.map(({ movie, score }) =>
        toMovieRecommendation(movie, score / maxScore)
      ));
      setView("recommendations");
    } finally {
      setLoading(false);
    }
  };

  const handleSimilarMovies = async (movieId: number) => {
    setLoading(true);
    setError(null);

    const movie = ALL_MOVIES.find(m => m.movie_id === movieId);
    setSimilarToMovie(movie?.title || `Movie ${movieId}`);

    try {
      const response = await getSimilarMovies(movieId, 15);
      setSimilarMovies(response.similar_movies);
      setView("similar");
    } catch {
      // Find similar movies by genre
      if (movie) {
        const movieGenres = new Set(movie.genres);
        const similar = ALL_MOVIES
          .filter(m => m.movie_id !== movieId)
          .map(m => {
            const commonGenres = m.genres.filter(g => movieGenres.has(g)).length;
            return { movie: m, similarity: commonGenres / Math.max(m.genres.length, movie.genres.length) };
          })
          .filter(({ similarity }) => similarity > 0)
          .sort((a, b) => b.similarity - a.similarity || b.movie.popularity - a.movie.popularity)
          .slice(0, 15);

        setSimilarMovies(similar.map(({ movie: m, similarity }) =>
          toMovieRecommendation(m, similarity)
        ));
      }
      setView("similar");
    } finally {
      setLoading(false);
    }
  };

  const resetColdStart = () => {
    setRatings({});
    setRecommendations([]);
    setView("coldstart");
  };

  const ratedCount = Object.values(ratings).filter(r => r > 0).length;

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-lg border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <button
              onClick={() => setView("home")}
              className="flex items-center gap-2 text-xl font-bold text-gray-900 dark:text-white hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
            >
              <Film className="w-7 h-7 text-blue-600" />
              <span>MovieRec</span>
            </button>

            <div className="hidden sm:block flex-1 max-w-md mx-8">
              <SearchBar onSearch={handleSearch} placeholder="Search movies..." />
            </div>

            <nav className="flex items-center gap-2">
              <button
                onClick={() => setView("coldstart")}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-md hover:shadow-lg"
              >
                <Sparkles className="w-4 h-4" />
                <span className="hidden sm:inline">Get Recommendations</span>
                <span className="sm:hidden">Rec</span>
              </button>
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Mobile search */}
        <div className="sm:hidden mb-6">
          <SearchBar onSearch={handleSearch} placeholder="Search movies..." />
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400">
            {error}
          </div>
        )}

        {/* Home View */}
        {view === "home" && (
          <div className="space-y-8">
            {/* Hero section */}
            <section className="text-center py-12 px-4">
              <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                Discover Your Next
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600"> Favorite Movie</span>
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto">
                Rate a few movies you love, and our AI will find personalized recommendations just for you.
              </p>
              <button
                onClick={() => setView("coldstart")}
                className="inline-flex items-center gap-2 px-6 py-3 text-lg font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
              >
                <Sparkles className="w-5 h-5" />
                Start Rating Movies
                <ArrowRight className="w-5 h-5" />
              </button>
            </section>

            {/* Genre filter */}
            <section>
              <div className="flex items-center gap-2 mb-4 overflow-x-auto pb-2">
                <TrendingUp className="w-5 h-5 text-gray-500 flex-shrink-0" />
                <span className="font-medium text-gray-700 dark:text-gray-300 flex-shrink-0">Popular in:</span>
                <div className="flex gap-2">
                  <button
                    onClick={() => loadPopularMovies()}
                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors flex-shrink-0 ${
                      !selectedGenre
                        ? "bg-blue-600 text-white"
                        : "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                    }`}
                  >
                    All
                  </button>
                  {genres.map(genre => (
                    <button
                      key={genre}
                      onClick={() => loadPopularMovies(genre)}
                      className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors flex-shrink-0 ${
                        selectedGenre === genre
                          ? "bg-blue-600 text-white"
                          : "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
                      }`}
                    >
                      {genre}
                    </button>
                  ))}
                </div>
              </div>
            </section>

            {/* Popular movies grid */}
            <section>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
                {selectedGenre ? `Popular ${selectedGenre} Movies` : "Popular Movies"}
                <span className="text-sm font-normal text-gray-500 ml-2">({popularMovies.length} movies)</span>
              </h2>
              {loading ? (
                <MovieGridSkeleton count={20} />
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 md:gap-6">
                  {popularMovies.map(movie => (
                    <MovieCard
                      key={movie.movie_id}
                      movie={movie}
                      onSimilarClick={handleSimilarMovies}
                      showScore={false}
                    />
                  ))}
                </div>
              )}
            </section>
          </div>
        )}

        {/* Cold Start View */}
        {view === "coldstart" && (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Rate Movies You Have Seen
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Rate at least 3 movies to get personalized recommendations
              </p>
            </div>

            {/* Progress indicator */}
            <div className="max-w-md mx-auto">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {ratedCount} of 3 minimum rated
                </span>
                <span className="text-sm font-medium text-blue-600">
                  {Math.min(ratedCount, 3) * 33}%
                </span>
              </div>
              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-600 to-purple-600 transition-all duration-300"
                  style={{ width: `${Math.min(ratedCount / 3 * 100, 100)}%` }}
                />
              </div>
            </div>

            {/* Movies to rate */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {SEED_MOVIES_FOR_RATING.map(movie => (
                <div
                  key={movie.movie_id}
                  className="bg-white dark:bg-gray-800 rounded-xl shadow-md p-4 flex gap-4"
                >
                  {/* Movie poster placeholder */}
                  <div
                    className="w-20 h-28 rounded-lg flex-shrink-0 flex items-center justify-center"
                    style={{
                      background: `linear-gradient(135deg, hsl(${(movie.movie_id * 137) % 360}, 70%, 60%), hsl(${((movie.movie_id * 137) + 60) % 360}, 70%, 40%))`,
                    }}
                  >
                    <Film className="w-8 h-8 text-white/30" />
                  </div>

                  {/* Movie info */}
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-gray-900 dark:text-white line-clamp-2 mb-1">
                      {movie.title}
                    </h3>
                    <p className="text-sm text-gray-500 mb-1">{movie.year}</p>
                    <div className="flex flex-wrap gap-1 mb-3">
                      {movie.genres.slice(0, 2).map(genre => (
                        <span
                          key={genre}
                          className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-xs rounded-full"
                        >
                          {genre}
                        </span>
                      ))}
                    </div>
                    <StarRating
                      value={ratings[movie.movie_id] || 0}
                      onChange={(rating) => handleRating(movie.movie_id, rating)}
                      size="md"
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Get recommendations button */}
            <div className="flex justify-center pt-4">
              <button
                onClick={handleGetRecommendations}
                disabled={ratedCount < 3 || loading}
                className="inline-flex items-center gap-2 px-8 py-4 text-lg font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    Getting Recommendations...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Get My Recommendations
                    <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Recommendations View */}
        {view === "recommendations" && (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Your Personalized Recommendations
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                Based on {ratedCount} movies you rated
              </p>
            </div>

            {loading ? (
              <MovieGridSkeleton count={20} />
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 md:gap-6">
                {recommendations.map((movie, index) => (
                  <div key={movie.movie_id} className="animate-fade-in" style={{ animationDelay: `${index * 50}ms` }}>
                    <MovieCard
                      movie={movie}
                      onSimilarClick={handleSimilarMovies}
                      showScore={true}
                    />
                  </div>
                ))}
              </div>
            )}

            <div className="flex justify-center pt-4">
              <button
                onClick={resetColdStart}
                className="inline-flex items-center gap-2 px-6 py-3 text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-xl hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <RefreshCw className="w-5 h-5" />
                Rate Different Movies
              </button>
            </div>
          </div>
        )}

        {/* Similar Movies View */}
        {view === "similar" && (
          <div className="space-y-8">
            <div>
              <button
                onClick={() => setView("home")}
                className="text-blue-600 hover:text-blue-700 mb-4 inline-flex items-center gap-1"
              >
                <ArrowRight className="w-4 h-4 rotate-180" />
                Back to Home
              </button>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                Movies Similar to
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-400">
                {similarToMovie}
              </p>
            </div>

            {loading ? (
              <MovieGridSkeleton count={15} />
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 md:gap-6">
                {similarMovies.map((movie, index) => (
                  <div key={movie.movie_id} className="animate-fade-in" style={{ animationDelay: `${index * 50}ms` }}>
                    <MovieCard
                      movie={movie}
                      onSimilarClick={handleSimilarMovies}
                      showScore={true}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Search Results View */}
        {view === "search" && (
          <div className="space-y-8">
            <div>
              <button
                onClick={() => setView("home")}
                className="text-blue-600 hover:text-blue-700 mb-4 inline-flex items-center gap-1"
              >
                <ArrowRight className="w-4 h-4 rotate-180" />
                Back to Home
              </button>
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                <Search className="w-8 h-8 inline mr-2" />
                Search Results
              </h2>
              <p className="text-gray-600 dark:text-gray-400">
                {searchResults.length} results for &ldquo;{searchQuery}&rdquo;
              </p>
            </div>

            {loading ? (
              <MovieGridSkeleton count={20} />
            ) : searchResults.length === 0 ? (
              <div className="text-center py-12">
                <Search className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                <p className="text-xl text-gray-500 dark:text-gray-400">
                  No movies found for &ldquo;{searchQuery}&rdquo;
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 md:gap-6">
                {searchResults.map((result, index) => {
                  const movie: MovieRecommendation = {
                    movie_id: result.movie_id,
                    title: result.title,
                    genres: result.genres,
                    year: result.year,
                    score: result.relevance_score,
                    avg_rating: null,
                    popularity: null,
                  };
                  return (
                    <div key={result.movie_id} className="animate-fade-in" style={{ animationDelay: `${index * 30}ms` }}>
                      <MovieCard
                        movie={movie}
                        onSimilarClick={handleSimilarMovies}
                        showScore={false}
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-16 py-8 border-t border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
              <Film className="w-5 h-5" />
              <span>MovieRec - AI-Powered Recommendations</span>
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-500">
              Built with Next.js, FastAPI, and Machine Learning
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
