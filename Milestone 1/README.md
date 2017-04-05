# Milestone 1 README

## Notebook

* *Milestone 1 - Submission* is an ipython notebook detailing steps completed for Milestone 1, including all data visualizations

## Scripts

* **get_detailed_movie_info** is a command line python script that takes the popular movie from TMDb and gets more detailed info on each movie from TMDd through the API (the most import of which is the `imdb_id`)

* **get_imdb_movie_info** is a command line python script that retreives information about a movie using IMDb - py from IMDb given it's `imdb_id` from the TMDb database

## Data Files

* **detailed_movie_data_imdb** has the IMDb movie details for all of the movies
* **detailed_movie_data_tmdb** has the TMDb movie details for all of the movies
* **popular_movie_data_tmdb** has the TMDb movie details provided by the TMDb popular movies api for all of the movies. It is less detailed than the other data file and does not include the `imdb_id`
* **tmdb_genre_encoding** is a file with the encoding of TMDb genres for each movie
