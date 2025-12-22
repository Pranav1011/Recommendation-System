"use client";

import { Star, Film, Calendar, Users } from "lucide-react";
import type { MovieRecommendation } from "@/types/api";
import { cn } from "@/lib/utils";

interface MovieCardProps {
  movie: MovieRecommendation;
  onSimilarClick?: (movieId: number) => void;
  showScore?: boolean;
  className?: string;
}

export function MovieCard({
  movie,
  onSimilarClick,
  showScore = true,
  className,
}: MovieCardProps) {
  // Generate a consistent color based on movie ID for the placeholder
  const hue = (movie.movie_id * 137) % 360;

  return (
    <div
      className={cn(
        "group relative bg-white dark:bg-gray-800 rounded-xl shadow-md hover:shadow-xl transition-all duration-300 overflow-hidden",
        className
      )}
    >
      {/* Poster placeholder with gradient */}
      <div
        className="aspect-[2/3] relative"
        style={{
          background: `linear-gradient(135deg, hsl(${hue}, 70%, 60%), hsl(${(hue + 60) % 360}, 70%, 40%))`,
        }}
      >
        <div className="absolute inset-0 flex items-center justify-center">
          <Film className="w-16 h-16 text-white/30" />
        </div>

        {/* Score badge */}
        {showScore && (
          <div className="absolute top-2 right-2 bg-black/70 backdrop-blur-sm text-white px-2 py-1 rounded-lg text-sm font-semibold">
            {(movie.score * 100).toFixed(0)}% match
          </div>
        )}

        {/* Hover overlay */}
        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
          {onSimilarClick && (
            <button
              onClick={() => onSimilarClick(movie.movie_id)}
              className="px-4 py-2 bg-white text-gray-900 rounded-lg font-medium hover:bg-gray-100 transition-colors"
            >
              Find Similar
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        <h3 className="font-semibold text-gray-900 dark:text-white line-clamp-2 min-h-[3rem]">
          {movie.title}
        </h3>

        {/* Genres */}
        <div className="flex flex-wrap gap-1 mt-2">
          {movie.genres.slice(0, 3).map((genre) => (
            <span
              key={genre}
              className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-xs rounded-full"
            >
              {genre}
            </span>
          ))}
          {movie.genres.length > 3 && (
            <span className="px-2 py-0.5 text-gray-400 text-xs">
              +{movie.genres.length - 3}
            </span>
          )}
        </div>

        {/* Metadata */}
        <div className="flex items-center gap-4 mt-3 text-sm text-gray-500 dark:text-gray-400">
          {movie.year && (
            <div className="flex items-center gap-1">
              <Calendar className="w-4 h-4" />
              <span>{movie.year}</span>
            </div>
          )}
          {movie.avg_rating && (
            <div className="flex items-center gap-1">
              <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
              <span>{movie.avg_rating.toFixed(1)}</span>
            </div>
          )}
          {movie.popularity && (
            <div className="flex items-center gap-1">
              <Users className="w-4 h-4" />
              <span>{(movie.popularity / 1000).toFixed(0)}k</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
