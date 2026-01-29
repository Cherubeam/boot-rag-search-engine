import json
import os

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0
MAX_CHUNK_SIZE = 4

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


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
