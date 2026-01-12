import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")


def load_movies() -> list[dict]:
    """Load movies from the JSON data file."""
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["movies"]