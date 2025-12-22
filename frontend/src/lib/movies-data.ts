// Comprehensive movie database with movies from 1960s to 2020s
// Using TMDB poster URLs with movie IDs

export interface MovieData {
  movie_id: number;
  title: string;
  genres: string[];
  year: number;
  tmdb_id: number; // For poster images
  avg_rating: number;
  popularity: number;
}

// TMDB image base URL - posters will be fetched from here
export const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500";

// Comprehensive movie list spanning decades
export const ALL_MOVIES: MovieData[] = [
  // 1970s Classics
  { movie_id: 858, title: "The Godfather", genres: ["Crime", "Drama"], year: 1972, tmdb_id: 238, avg_rating: 4.5, popularity: 95000 },
  { movie_id: 1221, title: "The Godfather Part II", genres: ["Crime", "Drama"], year: 1974, tmdb_id: 240, avg_rating: 4.4, popularity: 75000 },
  { movie_id: 1203, title: "Taxi Driver", genres: ["Crime", "Drama", "Thriller"], year: 1976, tmdb_id: 103, avg_rating: 4.2, popularity: 65000 },
  { movie_id: 1219, title: "Psycho", genres: ["Horror", "Mystery", "Thriller"], year: 1960, tmdb_id: 539, avg_rating: 4.3, popularity: 70000 },
  { movie_id: 1207, title: "Chinatown", genres: ["Crime", "Drama", "Mystery", "Thriller"], year: 1974, tmdb_id: 829, avg_rating: 4.2, popularity: 55000 },
  { movie_id: 904, title: "Jaws", genres: ["Adventure", "Drama", "Thriller"], year: 1975, tmdb_id: 578, avg_rating: 4.1, popularity: 80000 },
  { movie_id: 1196, title: "Star Wars: Episode V", genres: ["Action", "Adventure", "Sci-Fi"], year: 1980, tmdb_id: 1891, avg_rating: 4.3, popularity: 90000 },
  { movie_id: 260, title: "Star Wars: Episode IV", genres: ["Action", "Adventure", "Sci-Fi"], year: 1977, tmdb_id: 11, avg_rating: 4.2, popularity: 95000 },
  { movie_id: 1197, title: "Star Wars: Episode VI", genres: ["Action", "Adventure", "Sci-Fi"], year: 1983, tmdb_id: 1892, avg_rating: 4.1, popularity: 85000 },
  { movie_id: 1214, title: "Alien", genres: ["Horror", "Sci-Fi"], year: 1979, tmdb_id: 348, avg_rating: 4.2, popularity: 78000 },
  { movie_id: 1200, title: "Aliens", genres: ["Action", "Adventure", "Sci-Fi", "Thriller"], year: 1986, tmdb_id: 679, avg_rating: 4.1, popularity: 75000 },
  { movie_id: 1213, title: "Apocalypse Now", genres: ["Drama", "War"], year: 1979, tmdb_id: 28, avg_rating: 4.2, popularity: 60000 },
  { movie_id: 1240, title: "The Terminator", genres: ["Action", "Sci-Fi", "Thriller"], year: 1984, tmdb_id: 218, avg_rating: 4.0, popularity: 82000 },
  { movie_id: 589, title: "Terminator 2: Judgment Day", genres: ["Action", "Sci-Fi", "Thriller"], year: 1991, tmdb_id: 280, avg_rating: 4.2, popularity: 88000 },
  { movie_id: 1270, title: "Back to the Future", genres: ["Adventure", "Comedy", "Sci-Fi"], year: 1985, tmdb_id: 105, avg_rating: 4.1, popularity: 90000 },

  // 1980s Movies
  { movie_id: 1080, title: "Raiders of the Lost Ark", genres: ["Action", "Adventure"], year: 1981, tmdb_id: 85, avg_rating: 4.2, popularity: 88000 },
  { movie_id: 1291, title: "Indiana Jones and the Last Crusade", genres: ["Action", "Adventure"], year: 1989, tmdb_id: 89, avg_rating: 4.1, popularity: 80000 },
  { movie_id: 1961, title: "E.T. the Extra-Terrestrial", genres: ["Children", "Drama", "Fantasy", "Sci-Fi"], year: 1982, tmdb_id: 601, avg_rating: 4.0, popularity: 85000 },
  { movie_id: 1265, title: "Groundhog Day", genres: ["Comedy", "Fantasy", "Romance"], year: 1993, tmdb_id: 137, avg_rating: 4.0, popularity: 70000 },
  { movie_id: 1210, title: "Blade Runner", genres: ["Film-Noir", "Sci-Fi", "Thriller"], year: 1982, tmdb_id: 78, avg_rating: 4.1, popularity: 75000 },
  { movie_id: 1387, title: "Die Hard", genres: ["Action", "Thriller"], year: 1988, tmdb_id: 562, avg_rating: 4.0, popularity: 82000 },
  { movie_id: 2571, title: "The Matrix", genres: ["Action", "Sci-Fi", "Thriller"], year: 1999, tmdb_id: 603, avg_rating: 4.3, popularity: 95000 },
  { movie_id: 1193, title: "One Flew Over the Cuckoo's Nest", genres: ["Drama"], year: 1975, tmdb_id: 510, avg_rating: 4.3, popularity: 65000 },
  { movie_id: 7153, title: "Full Metal Jacket", genres: ["Drama", "War"], year: 1987, tmdb_id: 600, avg_rating: 4.1, popularity: 60000 },
  { movie_id: 1178, title: "The Shining", genres: ["Horror"], year: 1980, tmdb_id: 694, avg_rating: 4.2, popularity: 78000 },

  // 1990s Classics
  { movie_id: 318, title: "The Shawshank Redemption", genres: ["Crime", "Drama"], year: 1994, tmdb_id: 278, avg_rating: 4.5, popularity: 100000 },
  { movie_id: 296, title: "Pulp Fiction", genres: ["Comedy", "Crime", "Drama"], year: 1994, tmdb_id: 680, avg_rating: 4.4, popularity: 95000 },
  { movie_id: 356, title: "Forrest Gump", genres: ["Comedy", "Drama", "Romance"], year: 1994, tmdb_id: 13, avg_rating: 4.3, popularity: 92000 },
  { movie_id: 593, title: "The Silence of the Lambs", genres: ["Crime", "Horror", "Thriller"], year: 1991, tmdb_id: 274, avg_rating: 4.2, popularity: 85000 },
  { movie_id: 527, title: "Schindler's List", genres: ["Drama", "War"], year: 1993, tmdb_id: 424, avg_rating: 4.4, popularity: 80000 },
  { movie_id: 110, title: "Braveheart", genres: ["Action", "Drama", "War"], year: 1995, tmdb_id: 197, avg_rating: 4.1, popularity: 78000 },
  { movie_id: 50, title: "The Usual Suspects", genres: ["Crime", "Mystery", "Thriller"], year: 1995, tmdb_id: 629, avg_rating: 4.2, popularity: 75000 },
  { movie_id: 457, title: "The Fugitive", genres: ["Action", "Mystery", "Thriller"], year: 1993, tmdb_id: 5503, avg_rating: 4.0, popularity: 70000 },
  { movie_id: 480, title: "Jurassic Park", genres: ["Action", "Adventure", "Sci-Fi"], year: 1993, tmdb_id: 329, avg_rating: 4.0, popularity: 90000 },
  { movie_id: 1, title: "Toy Story", genres: ["Adventure", "Animation", "Children", "Comedy"], year: 1995, tmdb_id: 862, avg_rating: 4.0, popularity: 88000 },
  { movie_id: 150, title: "Apollo 13", genres: ["Adventure", "Drama", "IMAX"], year: 1995, tmdb_id: 568, avg_rating: 4.0, popularity: 68000 },
  { movie_id: 293, title: "Leon: The Professional", genres: ["Action", "Crime", "Drama", "Thriller"], year: 1994, tmdb_id: 101, avg_rating: 4.2, popularity: 80000 },
  { movie_id: 32, title: "Twelve Monkeys", genres: ["Drama", "Mystery", "Sci-Fi", "Thriller"], year: 1995, tmdb_id: 63, avg_rating: 4.0, popularity: 65000 },
  { movie_id: 592, title: "Batman", genres: ["Action", "Crime", "Fantasy"], year: 1989, tmdb_id: 268, avg_rating: 3.8, popularity: 72000 },
  { movie_id: 47, title: "Seven", genres: ["Crime", "Mystery", "Thriller"], year: 1995, tmdb_id: 807, avg_rating: 4.1, popularity: 82000 },
  { movie_id: 590, title: "Dances with Wolves", genres: ["Adventure", "Drama", "Western"], year: 1990, tmdb_id: 581, avg_rating: 4.0, popularity: 55000 },
  { movie_id: 364, title: "The Lion King", genres: ["Adventure", "Animation", "Children", "Drama"], year: 1994, tmdb_id: 8587, avg_rating: 4.1, popularity: 92000 },
  { movie_id: 588, title: "Aladdin", genres: ["Adventure", "Animation", "Children", "Comedy"], year: 1992, tmdb_id: 812, avg_rating: 4.0, popularity: 78000 },
  { movie_id: 595, title: "Beauty and the Beast", genres: ["Animation", "Children", "Fantasy", "Romance"], year: 1991, tmdb_id: 10020, avg_rating: 4.0, popularity: 76000 },
  { movie_id: 780, title: "Independence Day", genres: ["Action", "Adventure", "Sci-Fi", "Thriller"], year: 1996, tmdb_id: 602, avg_rating: 3.7, popularity: 75000 },
  { movie_id: 153, title: "Batman Forever", genres: ["Action", "Adventure", "Comedy", "Crime"], year: 1995, tmdb_id: 414, avg_rating: 3.2, popularity: 60000 },
  { movie_id: 377, title: "Speed", genres: ["Action", "Romance", "Thriller"], year: 1994, tmdb_id: 1637, avg_rating: 3.7, popularity: 68000 },
  { movie_id: 608, title: "Fargo", genres: ["Comedy", "Crime", "Drama", "Thriller"], year: 1996, tmdb_id: 275, avg_rating: 4.1, popularity: 65000 },
  { movie_id: 1617, title: "L.A. Confidential", genres: ["Crime", "Film-Noir", "Mystery", "Thriller"], year: 1997, tmdb_id: 2118, avg_rating: 4.1, popularity: 62000 },
  { movie_id: 1704, title: "Good Will Hunting", genres: ["Drama", "Romance"], year: 1997, tmdb_id: 489, avg_rating: 4.1, popularity: 72000 },
  { movie_id: 2028, title: "Saving Private Ryan", genres: ["Action", "Drama", "War"], year: 1998, tmdb_id: 857, avg_rating: 4.2, popularity: 85000 },
  { movie_id: 1721, title: "Titanic", genres: ["Drama", "Romance"], year: 1997, tmdb_id: 597, avg_rating: 3.9, popularity: 98000 },
  { movie_id: 1573, title: "Face/Off", genres: ["Action", "Crime", "Sci-Fi", "Thriller"], year: 1997, tmdb_id: 754, avg_rating: 3.8, popularity: 60000 },
  { movie_id: 2396, title: "Shakespeare in Love", genres: ["Comedy", "Drama", "Romance"], year: 1998, tmdb_id: 1934, avg_rating: 3.8, popularity: 55000 },
  { movie_id: 2997, title: "Being John Malkovich", genres: ["Comedy", "Drama", "Fantasy"], year: 1999, tmdb_id: 492, avg_rating: 4.0, popularity: 58000 },
  { movie_id: 2762, title: "The Sixth Sense", genres: ["Drama", "Mystery", "Thriller"], year: 1999, tmdb_id: 745, avg_rating: 4.0, popularity: 80000 },
  { movie_id: 2959, title: "Fight Club", genres: ["Action", "Crime", "Drama", "Thriller"], year: 1999, tmdb_id: 550, avg_rating: 4.3, popularity: 92000 },
  { movie_id: 3578, title: "Gladiator", genres: ["Action", "Adventure", "Drama"], year: 2000, tmdb_id: 98, avg_rating: 4.1, popularity: 88000 },
  { movie_id: 2858, title: "American Beauty", genres: ["Drama", "Romance"], year: 1999, tmdb_id: 14, avg_rating: 4.1, popularity: 75000 },
  { movie_id: 3147, title: "The Green Mile", genres: ["Crime", "Drama", "Fantasy", "Mystery"], year: 1999, tmdb_id: 497, avg_rating: 4.2, popularity: 78000 },
  { movie_id: 1198, title: "Raiders of the Lost Ark", genres: ["Action", "Adventure"], year: 1981, tmdb_id: 85, avg_rating: 4.2, popularity: 85000 },

  // 2000s Movies
  { movie_id: 4993, title: "The Lord of the Rings: The Fellowship of the Ring", genres: ["Adventure", "Fantasy"], year: 2001, tmdb_id: 120, avg_rating: 4.3, popularity: 95000 },
  { movie_id: 5952, title: "The Lord of the Rings: The Two Towers", genres: ["Adventure", "Fantasy"], year: 2002, tmdb_id: 121, avg_rating: 4.3, popularity: 90000 },
  { movie_id: 7153, title: "The Lord of the Rings: The Return of the King", genres: ["Adventure", "Fantasy"], year: 2003, tmdb_id: 122, avg_rating: 4.4, popularity: 92000 },
  { movie_id: 4226, title: "Memento", genres: ["Mystery", "Thriller"], year: 2000, tmdb_id: 77, avg_rating: 4.2, popularity: 72000 },
  { movie_id: 4963, title: "Ocean's Eleven", genres: ["Crime", "Thriller"], year: 2001, tmdb_id: 161, avg_rating: 3.9, popularity: 78000 },
  { movie_id: 5349, title: "Spider-Man", genres: ["Action", "Adventure", "Fantasy"], year: 2002, tmdb_id: 557, avg_rating: 3.8, popularity: 85000 },
  { movie_id: 4011, title: "Snatch", genres: ["Comedy", "Crime", "Thriller"], year: 2000, tmdb_id: 107, avg_rating: 4.0, popularity: 68000 },
  { movie_id: 4896, title: "Harry Potter and the Sorcerer's Stone", genres: ["Adventure", "Children", "Fantasy"], year: 2001, tmdb_id: 671, avg_rating: 4.0, popularity: 88000 },
  { movie_id: 5816, title: "Harry Potter and the Chamber of Secrets", genres: ["Adventure", "Children", "Fantasy"], year: 2002, tmdb_id: 672, avg_rating: 3.9, popularity: 82000 },
  { movie_id: 8636, title: "Harry Potter and the Prisoner of Azkaban", genres: ["Adventure", "Fantasy", "IMAX"], year: 2004, tmdb_id: 673, avg_rating: 4.0, popularity: 80000 },
  { movie_id: 5418, title: "Bourne Identity", genres: ["Action", "Mystery", "Thriller"], year: 2002, tmdb_id: 2501, avg_rating: 4.0, popularity: 75000 },
  { movie_id: 5445, title: "Minority Report", genres: ["Action", "Crime", "Mystery", "Sci-Fi", "Thriller"], year: 2002, tmdb_id: 180, avg_rating: 3.9, popularity: 70000 },
  { movie_id: 4979, title: "A Beautiful Mind", genres: ["Drama", "Romance"], year: 2001, tmdb_id: 453, avg_rating: 4.0, popularity: 72000 },
  { movie_id: 4878, title: "Donnie Darko", genres: ["Drama", "Mystery", "Sci-Fi", "Thriller"], year: 2001, tmdb_id: 141, avg_rating: 4.0, popularity: 65000 },
  { movie_id: 5481, title: "Catch Me If You Can", genres: ["Crime", "Drama"], year: 2002, tmdb_id: 640, avg_rating: 4.0, popularity: 75000 },
  { movie_id: 5952, title: "Chicago", genres: ["Comedy", "Crime", "Drama", "Musical"], year: 2002, tmdb_id: 1850, avg_rating: 3.8, popularity: 58000 },
  { movie_id: 6377, title: "Finding Nemo", genres: ["Adventure", "Animation", "Children", "Comedy"], year: 2003, tmdb_id: 12, avg_rating: 4.1, popularity: 88000 },
  { movie_id: 6539, title: "Pirates of the Caribbean: The Curse of the Black Pearl", genres: ["Action", "Adventure", "Comedy", "Fantasy"], year: 2003, tmdb_id: 22, avg_rating: 4.0, popularity: 90000 },
  { movie_id: 6874, title: "Kill Bill: Vol. 1", genres: ["Action", "Crime", "Thriller"], year: 2003, tmdb_id: 24, avg_rating: 4.0, popularity: 78000 },
  { movie_id: 7361, title: "Eternal Sunshine of the Spotless Mind", genres: ["Drama", "Romance", "Sci-Fi"], year: 2004, tmdb_id: 38, avg_rating: 4.1, popularity: 70000 },
  { movie_id: 8644, title: "The Incredibles", genres: ["Action", "Adventure", "Animation", "Children", "Comedy"], year: 2004, tmdb_id: 9806, avg_rating: 4.0, popularity: 82000 },
  { movie_id: 33794, title: "Batman Begins", genres: ["Action", "Crime", "Drama"], year: 2005, tmdb_id: 272, avg_rating: 4.0, popularity: 85000 },
  { movie_id: 40815, title: "Casino Royale", genres: ["Action", "Adventure", "Thriller"], year: 2006, tmdb_id: 36557, avg_rating: 4.0, popularity: 80000 },
  { movie_id: 45722, title: "300", genres: ["Action", "Fantasy", "War"], year: 2006, tmdb_id: 1271, avg_rating: 3.8, popularity: 75000 },
  { movie_id: 48516, title: "The Departed", genres: ["Crime", "Drama", "Thriller"], year: 2006, tmdb_id: 1422, avg_rating: 4.1, popularity: 78000 },
  { movie_id: 48780, title: "The Prestige", genres: ["Drama", "Mystery", "Sci-Fi", "Thriller"], year: 2006, tmdb_id: 1124, avg_rating: 4.1, popularity: 75000 },
  { movie_id: 48394, title: "Pan's Labyrinth", genres: ["Drama", "Fantasy", "Thriller", "War"], year: 2006, tmdb_id: 1417, avg_rating: 4.1, popularity: 65000 },
  { movie_id: 53121, title: "Ratatouille", genres: ["Animation", "Children", "Comedy"], year: 2007, tmdb_id: 2062, avg_rating: 4.0, popularity: 75000 },
  { movie_id: 54286, title: "No Country for Old Men", genres: ["Crime", "Drama", "Thriller"], year: 2007, tmdb_id: 6977, avg_rating: 4.0, popularity: 68000 },
  { movie_id: 54001, title: "Superbad", genres: ["Comedy"], year: 2007, tmdb_id: 8363, avg_rating: 3.8, popularity: 65000 },
  { movie_id: 57528, title: "Iron Man", genres: ["Action", "Adventure", "Sci-Fi"], year: 2008, tmdb_id: 1726, avg_rating: 4.0, popularity: 88000 },
  { movie_id: 58559, title: "The Dark Knight", genres: ["Action", "Crime", "Drama", "IMAX"], year: 2008, tmdb_id: 155, avg_rating: 4.3, popularity: 98000 },
  { movie_id: 59315, title: "WALL-E", genres: ["Adventure", "Animation", "Children", "Romance", "Sci-Fi"], year: 2008, tmdb_id: 10681, avg_rating: 4.1, popularity: 80000 },
  { movie_id: 60069, title: "Slumdog Millionaire", genres: ["Crime", "Drama", "Romance"], year: 2008, tmdb_id: 12405, avg_rating: 4.0, popularity: 72000 },
  { movie_id: 63082, title: "Avatar", genres: ["Action", "Adventure", "Sci-Fi", "IMAX"], year: 2009, tmdb_id: 19995, avg_rating: 3.9, popularity: 100000 },
  { movie_id: 68954, title: "Up", genres: ["Adventure", "Animation", "Children", "Drama"], year: 2009, tmdb_id: 14160, avg_rating: 4.1, popularity: 82000 },
  { movie_id: 69122, title: "District 9", genres: ["Mystery", "Sci-Fi", "Thriller"], year: 2009, tmdb_id: 17654, avg_rating: 4.0, popularity: 68000 },
  { movie_id: 72998, title: "Inglourious Basterds", genres: ["Action", "Drama", "War"], year: 2009, tmdb_id: 16869, avg_rating: 4.0, popularity: 78000 },

  // 2010s Movies
  { movie_id: 79132, title: "Inception", genres: ["Action", "Crime", "Drama", "Mystery", "Sci-Fi", "Thriller", "IMAX"], year: 2010, tmdb_id: 27205, avg_rating: 4.2, popularity: 95000 },
  { movie_id: 81845, title: "The Social Network", genres: ["Drama"], year: 2010, tmdb_id: 37799, avg_rating: 3.9, popularity: 72000 },
  { movie_id: 81834, title: "Toy Story 3", genres: ["Adventure", "Animation", "Children", "Comedy", "Fantasy", "IMAX"], year: 2010, tmdb_id: 10193, avg_rating: 4.1, popularity: 82000 },
  { movie_id: 81847, title: "Black Swan", genres: ["Drama", "Thriller"], year: 2010, tmdb_id: 44214, avg_rating: 4.0, popularity: 68000 },
  { movie_id: 85414, title: "The King's Speech", genres: ["Drama"], year: 2010, tmdb_id: 45269, avg_rating: 4.0, popularity: 65000 },
  { movie_id: 88125, title: "Drive", genres: ["Crime", "Drama", "Film-Noir", "Thriller"], year: 2011, tmdb_id: 64690, avg_rating: 3.9, popularity: 68000 },
  { movie_id: 86332, title: "Thor", genres: ["Action", "Adventure", "Drama", "Fantasy", "IMAX"], year: 2011, tmdb_id: 10195, avg_rating: 3.7, popularity: 75000 },
  { movie_id: 89745, title: "The Avengers", genres: ["Action", "Adventure", "Sci-Fi", "IMAX"], year: 2012, tmdb_id: 24428, avg_rating: 4.0, popularity: 95000 },
  { movie_id: 91529, title: "The Dark Knight Rises", genres: ["Action", "Adventure", "Crime", "IMAX"], year: 2012, tmdb_id: 49026, avg_rating: 4.0, popularity: 90000 },
  { movie_id: 96079, title: "Skyfall", genres: ["Action", "Adventure", "Thriller", "IMAX"], year: 2012, tmdb_id: 37724, avg_rating: 3.9, popularity: 82000 },
  { movie_id: 89864, title: "Django Unchained", genres: ["Action", "Drama", "Western"], year: 2012, tmdb_id: 68718, avg_rating: 4.1, popularity: 85000 },
  { movie_id: 99114, title: "Life of Pi", genres: ["Adventure", "Drama", "IMAX"], year: 2012, tmdb_id: 87827, avg_rating: 4.0, popularity: 72000 },
  { movie_id: 102125, title: "Wolf of Wall Street", genres: ["Comedy", "Crime", "Drama"], year: 2013, tmdb_id: 106646, avg_rating: 4.0, popularity: 82000 },
  { movie_id: 91542, title: "Frozen", genres: ["Adventure", "Animation", "Children", "Comedy", "Fantasy", "Musical", "Romance"], year: 2013, tmdb_id: 109445, avg_rating: 3.8, popularity: 92000 },
  { movie_id: 106782, title: "Her", genres: ["Drama", "Romance", "Sci-Fi"], year: 2013, tmdb_id: 152601, avg_rating: 4.0, popularity: 70000 },
  { movie_id: 106920, title: "Gravity", genres: ["Drama", "Sci-Fi", "Thriller", "IMAX"], year: 2013, tmdb_id: 49047, avg_rating: 3.8, popularity: 75000 },
  { movie_id: 109374, title: "Interstellar", genres: ["Adventure", "Drama", "Sci-Fi", "IMAX"], year: 2014, tmdb_id: 157336, avg_rating: 4.2, popularity: 95000 },
  { movie_id: 106916, title: "Guardians of the Galaxy", genres: ["Action", "Adventure", "Sci-Fi"], year: 2014, tmdb_id: 118340, avg_rating: 4.0, popularity: 88000 },
  { movie_id: 112552, title: "Whiplash", genres: ["Drama"], year: 2014, tmdb_id: 244786, avg_rating: 4.2, popularity: 72000 },
  { movie_id: 109487, title: "Gone Girl", genres: ["Drama", "Mystery", "Thriller"], year: 2014, tmdb_id: 210577, avg_rating: 4.0, popularity: 78000 },
  { movie_id: 116797, title: "The Grand Budapest Hotel", genres: ["Comedy", "Drama"], year: 2014, tmdb_id: 120467, avg_rating: 4.1, popularity: 70000 },
  { movie_id: 115713, title: "Ex Machina", genres: ["Drama", "Sci-Fi", "Thriller"], year: 2015, tmdb_id: 264660, avg_rating: 4.0, popularity: 72000 },
  { movie_id: 119145, title: "Kingsman: The Secret Service", genres: ["Action", "Adventure", "Comedy"], year: 2014, tmdb_id: 207703, avg_rating: 3.9, popularity: 75000 },
  { movie_id: 122882, title: "Mad Max: Fury Road", genres: ["Action", "Adventure", "Sci-Fi", "Thriller"], year: 2015, tmdb_id: 76341, avg_rating: 4.1, popularity: 85000 },
  { movie_id: 122886, title: "Star Wars: The Force Awakens", genres: ["Action", "Adventure", "Fantasy", "Sci-Fi", "IMAX"], year: 2015, tmdb_id: 140607, avg_rating: 3.9, popularity: 95000 },
  { movie_id: 122920, title: "The Martian", genres: ["Adventure", "Drama", "Sci-Fi", "IMAX"], year: 2015, tmdb_id: 286217, avg_rating: 4.0, popularity: 82000 },
  { movie_id: 134130, title: "The Revenant", genres: ["Adventure", "Drama", "Thriller"], year: 2015, tmdb_id: 281957, avg_rating: 3.9, popularity: 80000 },
  { movie_id: 135861, title: "Deadpool", genres: ["Action", "Adventure", "Comedy"], year: 2016, tmdb_id: 293660, avg_rating: 4.0, popularity: 90000 },
  { movie_id: 130634, title: "Arrival", genres: ["Drama", "Sci-Fi"], year: 2016, tmdb_id: 329865, avg_rating: 4.0, popularity: 75000 },
  { movie_id: 143385, title: "La La Land", genres: ["Comedy", "Drama", "Romance"], year: 2016, tmdb_id: 313369, avg_rating: 4.0, popularity: 78000 },
  { movie_id: 136864, title: "Hacksaw Ridge", genres: ["Drama", "History", "War"], year: 2016, tmdb_id: 324786, avg_rating: 4.1, popularity: 72000 },
  { movie_id: 162578, title: "Coco", genres: ["Adventure", "Animation", "Children", "Comedy"], year: 2017, tmdb_id: 354912, avg_rating: 4.2, popularity: 82000 },
  { movie_id: 168248, title: "Get Out", genres: ["Horror", "Mystery", "Thriller"], year: 2017, tmdb_id: 419430, avg_rating: 4.0, popularity: 78000 },
  { movie_id: 170875, title: "Logan", genres: ["Action", "Drama", "Sci-Fi"], year: 2017, tmdb_id: 263115, avg_rating: 4.1, popularity: 85000 },
  { movie_id: 168250, title: "Blade Runner 2049", genres: ["Mystery", "Sci-Fi"], year: 2017, tmdb_id: 335984, avg_rating: 4.0, popularity: 75000 },
  { movie_id: 172497, title: "Thor: Ragnarok", genres: ["Action", "Adventure", "Comedy", "Fantasy", "Sci-Fi"], year: 2017, tmdb_id: 284053, avg_rating: 4.0, popularity: 85000 },
  { movie_id: 179819, title: "Black Panther", genres: ["Action", "Adventure", "Sci-Fi"], year: 2018, tmdb_id: 284054, avg_rating: 3.9, popularity: 90000 },
  { movie_id: 176371, title: "A Quiet Place", genres: ["Drama", "Horror", "Sci-Fi"], year: 2018, tmdb_id: 447332, avg_rating: 3.9, popularity: 78000 },
  { movie_id: 179101, title: "Avengers: Infinity War", genres: ["Action", "Adventure", "Sci-Fi"], year: 2018, tmdb_id: 299536, avg_rating: 4.1, popularity: 95000 },
  { movie_id: 183611, title: "Spider-Man: Into the Spider-Verse", genres: ["Action", "Adventure", "Animation", "Sci-Fi"], year: 2018, tmdb_id: 324857, avg_rating: 4.2, popularity: 82000 },
  { movie_id: 185029, title: "Bohemian Rhapsody", genres: ["Drama"], year: 2018, tmdb_id: 424694, avg_rating: 3.9, popularity: 80000 },
  { movie_id: 193609, title: "Avengers: Endgame", genres: ["Action", "Adventure", "Sci-Fi"], year: 2019, tmdb_id: 299534, avg_rating: 4.1, popularity: 98000 },
  { movie_id: 187595, title: "Joker", genres: ["Crime", "Drama", "Thriller"], year: 2019, tmdb_id: 475557, avg_rating: 4.0, popularity: 92000 },
  { movie_id: 188675, title: "Parasite", genres: ["Comedy", "Drama", "Thriller"], year: 2019, tmdb_id: 496243, avg_rating: 4.3, popularity: 85000 },
  { movie_id: 190209, title: "1917", genres: ["Drama", "War"], year: 2019, tmdb_id: 530915, avg_rating: 4.1, popularity: 78000 },
  { movie_id: 189043, title: "Knives Out", genres: ["Comedy", "Crime", "Drama", "Mystery", "Thriller"], year: 2019, tmdb_id: 546554, avg_rating: 4.0, popularity: 80000 },
  { movie_id: 187593, title: "Once Upon a Time in Hollywood", genres: ["Comedy", "Drama"], year: 2019, tmdb_id: 466272, avg_rating: 3.9, popularity: 75000 },

  // 2020s Movies
  { movie_id: 193587, title: "Soul", genres: ["Adventure", "Animation", "Comedy", "Drama", "Fantasy"], year: 2020, tmdb_id: 508442, avg_rating: 4.1, popularity: 78000 },
  { movie_id: 207932, title: "Dune", genres: ["Action", "Adventure", "Drama", "Sci-Fi"], year: 2021, tmdb_id: 438631, avg_rating: 4.1, popularity: 88000 },
  { movie_id: 205587, title: "Spider-Man: No Way Home", genres: ["Action", "Adventure", "Fantasy", "Sci-Fi"], year: 2021, tmdb_id: 634649, avg_rating: 4.2, popularity: 95000 },
  { movie_id: 205181, title: "No Time to Die", genres: ["Action", "Adventure", "Thriller"], year: 2021, tmdb_id: 370172, avg_rating: 3.8, popularity: 80000 },
  { movie_id: 207702, title: "The Batman", genres: ["Action", "Crime", "Drama"], year: 2022, tmdb_id: 414906, avg_rating: 4.0, popularity: 85000 },
  { movie_id: 209163, title: "Everything Everywhere All at Once", genres: ["Action", "Adventure", "Comedy", "Fantasy", "Sci-Fi"], year: 2022, tmdb_id: 545611, avg_rating: 4.3, popularity: 82000 },
  { movie_id: 206647, title: "Top Gun: Maverick", genres: ["Action", "Drama"], year: 2022, tmdb_id: 361743, avg_rating: 4.2, popularity: 92000 },
  { movie_id: 213639, title: "Oppenheimer", genres: ["Drama", "History"], year: 2023, tmdb_id: 872585, avg_rating: 4.2, popularity: 90000 },
  { movie_id: 212587, title: "Barbie", genres: ["Adventure", "Comedy", "Fantasy"], year: 2023, tmdb_id: 346698, avg_rating: 3.8, popularity: 88000 },
  { movie_id: 217897, title: "Dune: Part Two", genres: ["Action", "Adventure", "Drama", "Sci-Fi"], year: 2024, tmdb_id: 693134, avg_rating: 4.3, popularity: 85000 },
];

// Get movies filtered by genre
export function getMoviesByGenre(genre: string | null, limit: number = 50): MovieData[] {
  let filtered = ALL_MOVIES;

  if (genre) {
    filtered = ALL_MOVIES.filter(m =>
      m.genres.some(g => g.toLowerCase() === genre.toLowerCase())
    );
  }

  // Sort by popularity and return limited results
  return filtered
    .sort((a, b) => b.popularity - a.popularity)
    .slice(0, limit);
}

// Get movies by decade
export function getMoviesByDecade(decade: number, limit: number = 20): MovieData[] {
  return ALL_MOVIES
    .filter(m => m.year >= decade && m.year < decade + 10)
    .sort((a, b) => b.popularity - a.popularity)
    .slice(0, limit);
}

// Search movies by title
export function searchMoviesByTitle(query: string, limit: number = 20): MovieData[] {
  const lowerQuery = query.toLowerCase();
  return ALL_MOVIES
    .filter(m => m.title.toLowerCase().includes(lowerQuery))
    .sort((a, b) => {
      // Prioritize exact matches
      const aExact = a.title.toLowerCase().startsWith(lowerQuery) ? 1 : 0;
      const bExact = b.title.toLowerCase().startsWith(lowerQuery) ? 1 : 0;
      if (aExact !== bExact) return bExact - aExact;
      return b.popularity - a.popularity;
    })
    .slice(0, limit);
}

// Get random diverse selection of movies
export function getRandomMovies(count: number = 50): MovieData[] {
  const shuffled = [...ALL_MOVIES].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
}

// Get TMDB poster URL for a movie
export function getPosterUrl(tmdbId: number): string {
  // Hardcoded poster paths for common movies (fetched from TMDB)
  const posterPaths: Record<number, string> = {
    238: "/3bhkrj58Vtu7enYsRolD1fZdja1.jpg", // Godfather
    240: "/hek3koDUyRQq7bkV3Xu7AREtQP.jpg", // Godfather 2
    278: "/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg", // Shawshank
    680: "/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg", // Pulp Fiction
    13: "/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg", // Forrest Gump
    274: "/uS9m8OBk1A8eM9I042bx8XXpqAq.jpg", // Silence of the Lambs
    11: "/6FfCtAuVAW8XJjZ7eWeLibRLWTw.jpg", // Star Wars IV
    329: "/oU7Oq2kFAAlGqbU4VoAE36g4hoI.jpg", // Jurassic Park
    862: "/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg", // Toy Story
    603: "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg", // Matrix
    120: "/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg", // LOTR Fellowship
    121: "/5VTN0pR8gcqV3EPUHHfMGnJYN9L.jpg", // LOTR Two Towers
    122: "/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg", // LOTR Return
    155: "/qJ2tW6WMUDux911r6m7haRef0WH.jpg", // Dark Knight
    27205: "/edv5CZvWj09upOsy2Y6IwDhK8bt.jpg", // Inception
    550: "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg", // Fight Club
    157336: "/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg", // Interstellar
    19995: "/kyeqWdyUXW608qlYkRqosgbbJyK.jpg", // Avatar
    597: "/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg", // Titanic
    24428: "/RYMX2wcKCBAr24UyPD7xwmjaTn.jpg", // Avengers
    299536: "/7WsyChQLEftFiDOVTGkv3hFpyyt.jpg", // Infinity War
    299534: "/or06FN3Dka5tukK1e9sl16pB3iy.jpg", // Endgame
    475557: "/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg", // Joker
    496243: "/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg", // Parasite
    438631: "/d5NXSklXo0qyIYkgV94XAgMIckC.jpg", // Dune
    634649: "/1g0dhYtq4irTY1GPXvft6k4YLjm.jpg", // Spider-Man No Way Home
    545611: "/w3LxiVYdWWRvEVdn5RYq6jIqkb1.jpg", // Everything Everywhere
    361743: "/62HCnUTziyWcpDaBO2i1DX17ljH.jpg", // Top Gun Maverick
    872585: "/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg", // Oppenheimer
    346698: "/iuFNMS8U5cb6xfzi51Dbkovj7vM.jpg", // Barbie
    693134: "/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg", // Dune 2
    98: "/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg", // Gladiator
    101: "/yI6X2cCM5YPJtxMhUd3dPGqDAhw.jpg", // Leon
    807: "/6yoghtyTpznpBik8EngEmJskVUO.jpg", // Se7en
    77: "/yuNs09hvpHVU1cBTCAk9zxsL2oW.jpg", // Memento
    78: "/63N9uy8nd9j7Eog2axPQ8lbr3Wj.jpg", // Blade Runner
    562: "/yFihWxQcmqcaBR31QM6Y8gT6aYV.jpg", // Die Hard
    8587: "/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg", // Lion King
    12: "/eHuGQ10FUzK1mdOY69wF5pGgEf5.jpg", // Finding Nemo
    22: "/z8onk7LV9Mmw6zKz4hT6pzzvmvl.jpg", // Pirates Caribbean
    24: "/v7TKYjCTFOpX4DoNfDOv7wEXUqj.jpg", // Kill Bill
    38: "/5MwkWH9tYHv3mV9OdYTMR5qreIz.jpg", // Eternal Sunshine
  };

  if (posterPaths[tmdbId]) {
    return `${TMDB_IMAGE_BASE}${posterPaths[tmdbId]}`;
  }

  // Return a placeholder or empty for unknown movies
  return "";
}

// Seed movies for cold start rating (diverse selection)
export const SEED_MOVIES_FOR_RATING: MovieData[] = [
  ALL_MOVIES.find(m => m.movie_id === 318)!, // Shawshank
  ALL_MOVIES.find(m => m.movie_id === 296)!, // Pulp Fiction
  ALL_MOVIES.find(m => m.movie_id === 356)!, // Forrest Gump
  ALL_MOVIES.find(m => m.movie_id === 2571)!, // Matrix
  ALL_MOVIES.find(m => m.movie_id === 858)!, // Godfather
  ALL_MOVIES.find(m => m.movie_id === 260)!, // Star Wars IV
  ALL_MOVIES.find(m => m.movie_id === 4993)!, // LOTR Fellowship
  ALL_MOVIES.find(m => m.movie_id === 58559)!, // Dark Knight
  ALL_MOVIES.find(m => m.movie_id === 79132)!, // Inception
  ALL_MOVIES.find(m => m.movie_id === 1)!, // Toy Story
  ALL_MOVIES.find(m => m.movie_id === 109374)!, // Interstellar
  ALL_MOVIES.find(m => m.movie_id === 188675)!, // Parasite
  ALL_MOVIES.find(m => m.movie_id === 63082)!, // Avatar
  ALL_MOVIES.find(m => m.movie_id === 207932)!, // Dune
  ALL_MOVIES.find(m => m.movie_id === 209163)!, // Everything Everywhere
].filter(Boolean);
