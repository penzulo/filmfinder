from os import getenv
from os.path import isfile
from typing import List, Optional

from dotenv import load_dotenv

from models.MovieRecommender import MovieRecommender


def main() -> None:
    load_dotenv()
    api_key: Optional[str] = getenv("API_KEY")
    if api_key is None:
        raise ValueError("API key not found in environment variables.")

    recommender: MovieRecommender = MovieRecommender(api_key)

    movie_titles: List[str] = [
        "Inception",
        "The Matrix",
        "Interstellar",
        "The Dark Knight",
        "Pulp Fiction",
        "Forrest Gump",
        "The Shawshank Redemption",
        "Fight Club",
        "The Godfather",
        "The Godfather Part II",
        "The Lord of the Rings: The Fellowship of the Ring",
        "The Lord of the Rings: The Two Towers",
        "The Lord of the Rings: The Return of the King",
        "Star Wars: Episode IV - A New Hope",
        "Star Wars: Episode V - The Empire Strikes Back",
        "Star Wars: Episode VI - Return of the Jedi",
        "The Avengers",
        "Avengers: Endgame",
        "Iron Man",
        "Iron Man 2",
        "Iron Man 3",
        "Black Panther",
        "Captain America: The First Avenger",
        "Captain America: The Winter Soldier",
        "Captain America: Civil War",
        "Guardians of the Galaxy",
        "Guardians of the Galaxy Vol. 2",
        "Thor",
        "Thor: Ragnarok",
        "Spider-Man: Homecoming",
        "Spider-Man: Far From Home",
        "Doctor Strange",
        "Ant-Man",
        "Ant-Man and the Wasp",
        "The Lion King",
        "Aladdin",
        "Frozen",
        "Frozen II",
        "Beauty and the Beast",
        "Cinderella",
        "Mulan",
        "Toy Story",
        "Toy Story 2",
        "Toy Story 3",
        "Toy Story 4",
        "Finding Nemo",
        "Finding Dory",
        "Up",
        "Wall-E",
        "Coco",
        "Inside Out",
        "Soul",
    ]
    if not isfile("movies_dataset.json"):
        recommender.enrich_dataset(movie_titles)
        recommender.save_dataset("movies_dataset.json")
    else:
        recommender.load_dataset("movies_dataset.json")

    recommender.generate_similarity_matrix()

    while True:
        input_movie: str = input("\nEnter a movie title (or 'quit' to exit): ")
        if not isinstance(input_movie, str):
            print("Invalid input. Please enter a valid movie title or 'quit' to exit.")
            continue

        if input_movie.lower() == "quit":
            break

        if input_movie not in movie_titles:
            print("Movie not found in the dataset.")
            continue
        try:
            recommendations = recommender.recommend(input_movie, n=5)
            print(f"Movies similar to '{input_movie}':")
            for i, movie in enumerate(recommendations, start=1):
                print(f"{i}. {movie}")
        except KeyError:
            print(f"Movie '{input_movie}' not found in the dataset.")
            print("Available movies:", " ".join(movie_titles))


if __name__ == "__main__":
    main()
