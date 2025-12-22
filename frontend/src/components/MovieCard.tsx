"use client";

import { useState } from "react";
import Image from "next/image";
import { Star, Film, Calendar, Users } from "lucide-react";
import type { MovieRecommendation } from "@/types/api";
import { cn } from "@/lib/utils";

// TMDB poster paths for popular movies
const POSTER_PATHS: Record<number, string> = {
  // 1970s-80s
  858: "/3bhkrj58Vtu7enYsRolD1fZdja1.jpg", // Godfather
  1221: "/hek3koDUyRQq7bkV3Xu7AREtQP.jpg", // Godfather 2
  1203: "/ekstpH614fwDX8DUln1a2Opz0N8.jpg", // Taxi Driver
  1219: "/yz4QVqPx3h1hD1DfqqQkCq3rmxW.jpg", // Psycho
  904: "/lxM6kqilAdpdhqUl2biYp5frUxE.jpg", // Jaws
  260: "/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg", // Star Wars IV
  1196: "/nNAeTmF4CtdSgMDplXTDPOpYzsX.jpg", // Star Wars V
  1197: "/jQYlydvHm3kUix1f8prMucrplhm.jpg", // Star Wars VI
  1214: "/vfrQk5IPloGg1v9Rzbh2Eg3VGyM.jpg", // Alien
  1200: "/r1x5JGpyqZU8PYhbs4UcrO1Xb6x.jpg", // Aliens
  1213: "/gQB8Y5RCMkv2zwzFHbUJX3kAhvA.jpg", // Apocalypse Now
  1240: "/qvktm0BHcnmDpul4Hz01GIazWPr.jpg", // Terminator
  589: "/5M0j0B18abtBI5gi2RhfjjurTqb.jpg", // Terminator 2
  1270: "/fNOH9f1aA7XRTzl1sAOx9iF553Q.jpg", // Back to the Future
  1080: "/ceG9VzoRAVGwivFU403Wc3AHRys.jpg", // Raiders
  1291: "/4p1N2Qrt8j0H8xMHMHvtRxv9weZ.jpg", // Last Crusade
  1961: "/an0nD6uq6bLxj4NBT3frkirNBUo.jpg", // ET
  1210: "/63N9uy8nd9j7Eog2axPQ8lbr3Wj.jpg", // Blade Runner
  1387: "/yFihWxQcmqcaBR31QM6Y8gT6aYV.jpg", // Die Hard
  1193: "/3jcbDmRFiQ83drXNOvRDeKHxS0C.jpg", // Cuckoo's Nest
  1178: "/b6ko0IKC8MdYBBPkkA1aBPLe2yz.jpg", // The Shining

  // 1990s
  318: "/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg", // Shawshank
  296: "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg", // Pulp Fiction
  356: "/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg", // Forrest Gump
  593: "/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg", // Silence of Lambs
  527: "/sF1U4EUQS8YHUYjNl3pMGNIQyr0.jpg", // Schindler's List
  110: "/doiUtOHzcxXFl0GVQ2n4S4U8Ob3.jpg", // Braveheart
  50: "/bUPmtQzrRhzqYySeiMpv7GurAfm.jpg", // Usual Suspects
  480: "/oU7Oq2kFAAlGqbU4VoAE36g4hoI.jpg", // Jurassic Park
  1: "/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg", // Toy Story
  293: "/yI6X2cCM5YPJtxMhUd3dPGqDAhw.jpg", // Leon
  47: "/6yoghtyTpznpBik8EngEmJskVUO.jpg", // Se7en
  364: "/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg", // Lion King
  588: "/ru0EB9qqTEGMTqLV6vvQINGjfSC.jpg", // Aladdin
  595: "/9BQqngPfwpeAfK7c2H3cwIFWIVR.jpg", // Beauty and Beast
  608: "/rt7cpEr1uP6RTZykBFhBTtPVt6p.jpg", // Fargo
  1617: "/5AO1TUff6v4Xozvv9dMq0bdFohi.jpg", // LA Confidential
  1704: "/bABCBKYBK7A5G1x0FzoeoNfuj2b.jpg", // Good Will Hunting
  2028: "/uqx37cS8cpHg8U35f9U5IBlrCV3.jpg", // Saving Private Ryan
  1721: "/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg", // Titanic
  2762: "/fLyhl5VXUUUT3FgkHrqNqv7ioBc.jpg", // Sixth Sense
  2959: "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg", // Fight Club
  2571: "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg", // Matrix
  2858: "/wby9315QzVKdW9BonAefg8jGTTb.jpg", // American Beauty
  3147: "/velWPhVMQeQKcxggNEU8YmIo52R.jpg", // Green Mile
  3578: "/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg", // Gladiator

  // 2000s
  4993: "/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg", // LOTR Fellowship
  5952: "/5VTN0pR8gcqV3EPUHHfMGnJYN9L.jpg", // LOTR Two Towers
  7153: "/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg", // LOTR Return
  4226: "/yuNs09hvpHVU1cBTCAk9zxsL2oW.jpg", // Memento
  4963: "/o0h76DVXvk5OKjmNez5YY0GODC2.jpg", // Ocean's Eleven
  5349: "/gh4cZbhZxyTbgxQPxD0dOudNPTn.jpg", // Spider-Man
  4011: "/56mOJth6DJ6JhgoE2jtpilVqJO.jpg", // Snatch
  4896: "/wuMc08IPKEatf9rnMNXvIDxqP4W.jpg", // Harry Potter 1
  5816: "/sdEOH0992YZ0QSxgXNIGLq1ToUi.jpg", // Harry Potter 2
  8636: "/aWxwnYoe8p2d2fcxOqtvAtJ72Rw.jpg", // Harry Potter 3
  5418: "/bXQIL36VQdzJ69lcjQR1WQzJqQR.jpg", // Bourne Identity
  5445: "/h3lpltSn7Rj1eYTPQO1lYGdw4Bz.jpg", // Minority Report
  4979: "/4SFqHDZ1NvWdysucWbgnYlobdxC.jpg", // Beautiful Mind
  4878: "/fhQoQfejY1hUcwyuLgpBrYs6uFt.jpg", // Donnie Darko
  5481: "/sdJDuYEF0dFnWt6VpXswdT0O9FA.jpg", // Catch Me If You Can
  6377: "/eHuGQ10FUzK1mdOY69wF5pGgEf5.jpg", // Finding Nemo
  6539: "/z8onk7LV9Mmw6zKz4hT6pzzvmvl.jpg", // Pirates Caribbean
  6874: "/v7TKYjCTFOpX4DoNfDOv7wEXUqj.jpg", // Kill Bill
  7361: "/5MwkWH9tYHv3mV9OdYTMR5qreIz.jpg", // Eternal Sunshine
  8644: "/2LqaLgk4Z226KkgPJuiOQ58wvrm.jpg", // Incredibles
  33794: "/8RW2runSEc34IwKN2D1aPcJd2UL.jpg", // Batman Begins
  40815: "/36bPbSLhIaJIScQzXZstXSqnnVi.jpg", // Casino Royale
  45722: "/7IJ7F8tX7IAkpUdaGovOBJqORnJ.jpg", // 300
  48516: "/jyAgiqVSx5fl0NNj7WoGGKweXrL.jpg", // Departed
  48780: "/tRNlZbgNCNOpLpbPEz5L8G8A0JN.jpg", // Prestige
  48394: "/sISHaXEgKjO3v9qSihScFPtOOJF.jpg", // Pan's Labyrinth
  53121: "/t3vaWRPSf6WjDSamIkKDs1iQWna.jpg", // Ratatouille
  54286: "/bj1v6YKF8yHqA489VFfnQvOJpnc.jpg", // No Country
  57528: "/78lPtwv72eTNqFW9COBYI0dWDJa.jpg", // Iron Man
  58559: "/qJ2tW6WMUDux911r6m7haRef0WH.jpg", // Dark Knight
  59315: "/hbhFnRzzg6ZDmm8YAmxBnQpQIPh.jpg", // WALL-E
  63082: "/kyeqWdyUXW608qlYkRqosgbbJyK.jpg", // Avatar
  68954: "/vpbaStTvWwEp5KNGWyPFEOlqGEz.jpg", // Up
  72998: "/7sfbEnaARXDDhKm0CZ7D7uc2sbo.jpg", // Inglourious Basterds

  // 2010s
  79132: "/edv5CZvWj09upOsy2Y6IwDhK8bt.jpg", // Inception
  81845: "/n0ybibhJtQ5icDqTp8eRytcIHJx.jpg", // Social Network
  81834: "/mMltbSxwEdNE4Cv8QYLpzkHWTDo.jpg", // Toy Story 3
  81847: "/nVL7OrFyxB5aECEGb5D1JqS7IRg.jpg", // Black Swan
  89745: "/RYMX2wcKCBAr24UyPD7xwmjaTn.jpg", // Avengers
  91529: "/hr0L2aueqlP2BYUblTTjmtn0hw4.jpg", // Dark Knight Rises
  96079: "/9BQqngPfwpeAfK7c2H3cwIFWIVR.jpg", // Skyfall
  89864: "/7oWY8VDWW7thTzWh3OKYRkWUlD5.jpg", // Django
  102125: "/pWHf4khOloNVfCxscsXFj3jj6gP.jpg", // Wolf of Wall Street
  91542: "/kgwjIb2JDHRhNk13lmSxiClFjVk.jpg", // Frozen
  106782: "/yk4J4aewWYNiBhD49WD7UaBBn37.jpg", // Her
  109374: "/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg", // Interstellar
  106916: "/r7vmZjiyZw9rpJMQJdXpjgiCOk9.jpg", // Guardians Galaxy
  112552: "/lIv1QinFqz4dlp5U4lQ6HaiskOZ.jpg", // Whiplash
  109487: "/lv5xShBIDPe7m6u4iPTu76uzWqx.jpg", // Gone Girl
  116797: "/eWdyYQreja6JGCzqHWXpWHDrrPo.jpg", // Grand Budapest
  115713: "/d9na2fI0wET56baR82LkS6VEA1a.jpg", // Ex Machina
  122882: "/8tZYtuWezp8JbcsvHYO0O46tFbo.jpg", // Mad Max Fury Road
  122886: "/wqnLdLLarWwDSK4AIEqA1wRtOTQ.jpg", // Force Awakens
  122920: "/5BHuvQ6p9kfc091Z8RiFNhCwL4b.jpg", // Martian
  134130: "/ji3ecJphATlVgWNY0B0RVXZizdf.jpg", // Revenant
  135861: "/inVq3FRqcYIRl2la8iZikYYxFNR.jpg", // Deadpool
  130634: "/x2FJsf1ElAgr63Y3PNPtJrcmpoe.jpg", // Arrival
  143385: "/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg", // La La Land
  162578: "/gGEsBPAijhVUFoiNpgZXqRVWJt2.jpg", // Coco
  168248: "/qbaIViX3tD2hUK0P0qIlMgjGvkr.jpg", // Get Out
  170875: "/fnbjcRDYn6YviCcePDnGdyAkYsB.jpg", // Logan
  168250: "/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg", // Blade Runner 2049
  172497: "/rzRwTcFvttcN1ZpX2xv4j3tSdJu.jpg", // Thor Ragnarok
  179819: "/uxzzxijgPIY7slzFvMotPv8wjKA.jpg", // Black Panther
  176371: "/nAU74GmpUk7t5iklEp3bufwDq4n.jpg", // Quiet Place
  179101: "/7WsyChQLEftFiDOVTGkv3hFpyyt.jpg", // Infinity War
  183611: "/iiZZdoQBEYBv6id8su7ImL0oCbD.jpg", // Spider-Verse
  185029: "/lHu1wtNaczFPGFDTrjCSzeLPTKN.jpg", // Bohemian Rhapsody
  193609: "/or06FN3Dka5tukK1e9sl16pB3iy.jpg", // Endgame
  187595: "/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg", // Joker
  188675: "/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg", // Parasite
  190209: "/iZvSfgRt2sSB2amoJzjoBTJRfSu.jpg", // 1917
  189043: "/pThyQovXQrw2m0s9x82twj48Jq4.jpg", // Knives Out
  187593: "/8j58iEBw9pOXFD2L0nt0ZXeHviB.jpg", // Once Upon Hollywood

  // 2020s
  193587: "/hm58Jw4Lw8OIeECIq5qyPYhAeRJ.jpg", // Soul
  207932: "/d5NXSklXo0qyIYkgV94XAgMIckC.jpg", // Dune
  205587: "/1g0dhYtq4irTY1GPXvft6k4YLjm.jpg", // Spider-Man No Way Home
  205181: "/iUgygt3fscRoKWCV1d0C7FbM9TP.jpg", // No Time to Die
  207702: "/74xTEgt7R36Fvez9TiTb06xXHQH.jpg", // The Batman
  209163: "/w3LxiVYdWWRvEVdn5RYq6jIqkb1.jpg", // Everything Everywhere
  206647: "/62HCnUTziyWcpDaBO2i1DX17ljH.jpg", // Top Gun Maverick
  213639: "/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg", // Oppenheimer
  212587: "/iuFNMS8U5cb6xfzi51Dbkovj7vM.jpg", // Barbie
  217897: "/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg", // Dune 2
};

const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500";

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
  const [imageError, setImageError] = useState(false);

  // Generate a consistent color based on movie ID for the placeholder
  const hue = (movie.movie_id * 137) % 360;

  // Get poster URL if available
  const posterPath = POSTER_PATHS[movie.movie_id];
  const posterUrl = posterPath ? `${TMDB_IMAGE_BASE}${posterPath}` : null;

  return (
    <div
      className={cn(
        "group relative bg-white dark:bg-gray-800 rounded-xl shadow-md hover:shadow-xl transition-all duration-300 overflow-hidden",
        className
      )}
    >
      {/* Poster */}
      <div className="aspect-[2/3] relative">
        {posterUrl && !imageError ? (
          <Image
            src={posterUrl}
            alt={movie.title}
            fill
            className="object-cover"
            sizes="(max-width: 640px) 50vw, (max-width: 768px) 33vw, (max-width: 1024px) 25vw, 20vw"
            onError={() => setImageError(true)}
          />
        ) : (
          <div
            className="w-full h-full flex items-center justify-center"
            style={{
              background: `linear-gradient(135deg, hsl(${hue}, 70%, 60%), hsl(${(hue + 60) % 360}, 70%, 40%))`,
            }}
          >
            <Film className="w-16 h-16 text-white/30" />
          </div>
        )}

        {/* Score badge */}
        {showScore && movie.score && (
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
