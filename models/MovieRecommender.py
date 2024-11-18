from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import requests
from numpy.typing import NDArray
from pandas import DataFrame, read_json
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(repr=True, frozen=True)
class Movie:
    """
    Represents a movie with the following attributes:

    - `title`: The title of the movie.
    - `year`: The year the movie was released.
    - `genre`: The genre(s) of the movie.
    - `director`: The director(s) of the movie.
    - `plot`: A brief plot summary of the movie.
    - `rating`: The rating of the movie, typically on a scale of 0.0 to 10.0.
    """

    title: str
    year: int
    genre: str
    director: str
    plot: str
    rating: float


@dataclass(repr=True)
class MovieRecommender:
    """
    Represents a movie recommender system that can fetch movie details, enrich a dataset, generate a similarity matrix, and provide movie recommendations.

    The `MovieRecommender` class has the following methods:

    - `fetch_movie_details(movie_title: str) -> Optional[Movie]`: Fetches movie details from the OMDb API for the given movie title.
    - `enrich_dataset(movie_title: List[str]) -> None`: Enriches the movie dataset by fetching movie details for the given list of movie titles.
    - `generate_similarity_matrix() -> None`: Generates a cosine similarity matrix based on the movie genres.
    - `recommend(movie_title: str, n: int = 5) -> Optional[List[str]]`: Provides a list of n recommended movie titles based on the cosine similarity matrix.
    - `save_dataset(file_path: str) -> None`: Saves the movie dataset to the specified file path in JSON format.
    - `load_dataset(file_path: str) -> None`: Loads the movie dataset from the specified file path.
    """

    api_key: str
    movie_data: DataFrame = field(default_factory=DataFrame)
    cosine_sim_matrix: NDArray = field(default=None)

    def __init__(self, api_key: str):
        self.api_key: str = api_key
        self.movie_data: DataFrame = DataFrame()
        self.cosine_sim_matrix = None

    def fetch_movie_details(self, movie_title: str) -> Optional[Movie]:
        """
        Fetches movie details from the OMDb API for the given movie title.

        Args:
            movie_title (str): The title of the movie to fetch details for.

        Returns:
            Optional[Movie]: A `Movie` object containing the fetched movie details, or `None` if the movie was not found or an error occurred.
        """
        url: str = f"http://www.omdbapi.com/"
        params: Dict[str, str] = {"apikey": self.api_key, "t": movie_title}

        try:
            response: requests.Response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("Response") == "True":
                return Movie(
                    title=data.get("Title"),
                    year=int(data.get("Year")),
                    genre=data.get("Genre"),
                    director=data.get("Director"),
                    plot=data.get("Plot"),
                    rating=(
                        float(data.get("imdbRating"))
                        if not data.get("imdbRating") == "N/A"
                        else 0.0
                    ),
                )
            else:
                print(f"Movie not found: {movie_title}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching movie details: {e}")
            return None

    def enrich_dataset(self, movie_title: List[str]) -> None:
        """
        Enriches the movie dataset by fetching movie details from the OMDb API for the given list of movie titles.

        Args:
            movie_title (List[str]): A list of movie titles to fetch details for.

        Returns:
            None
        """
        enriched_data: List[Optional[Movie]] = [
            self.fetch_movie_details(title) for title in movie_title if title
        ]
        self.movie_data = DataFrame(
            [movie for movie in enriched_data if movie is not None]
        )

    def generate_similarity_matrix(self) -> None:
        if self.movie_data.empty:
            raise ValueError(
                "Movie data is empty. Please enrich the movie dataset first."
            )

        # self.movie_data["genre"] = self.movie_data["genre"].str.split(", ")
        genre_matrix: Union[NDArray, spmatrix] = CountVectorizer().fit_transform(
            self.movie_data["genre"].fillna("")
        )
        self.cosine_sim_matrix = cosine_similarity(genre_matrix, genre_matrix)

    def recommend(self, movie_title: str, n: int = 5) -> Optional[List[str]]:
        if self.cosine_sim_matrix is None:
            raise ValueError(
                "Similarity matrix is not generated. Please generate it first."
            )

        try:
            movie_index = self.movie_data[
                self.movie_data["title"] == movie_title
            ].index[0]
            similarity_scores = list(enumerate(self.cosine_sim_matrix[movie_index]))
            similarity_scores = sorted(
                similarity_scores, key=lambda x: x[1], reverse=True
            )
            top_indices = [i[0] for i in similarity_scores[1 : n + 1]]
            return self.movie_data.iloc[top_indices]["title"].tolist()
        except IndexError:
            print(f"Movie not found: {movie_title}")
            return None

    def save_dataset(self, file_path: str) -> None:
        """
        Saves the movie dataset to the specified file path in JSON format.

        Args:
            file_path (str): The file path to save the dataset to.

        Returns:
            None
        """
        self.movie_data.to_json(file_path, orient="records", indent=4)
        print(f"Dataset saved to: {file_path}")

    def load_dataset(self, file_path: str) -> None:
        self.movie_data = DataFrame(read_json(file_path, orient="records"))
        print(f"Dataset loaded from: {file_path}")
