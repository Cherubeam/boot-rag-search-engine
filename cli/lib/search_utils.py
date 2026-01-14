import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_movies() -> list[dict]:
    """Load movies from the JSON data file."""
    with open(MOVIES_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["movies"]


def load_stopwords() -> list[str]:
    """Load stopwords from a predefined list."""
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as file:
        data = file.read().splitlines()
    return data