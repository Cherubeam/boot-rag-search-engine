import string

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """Search movies by title using a simple keyword match.

    Args:
        query (str): The search query.
        limit (int): The maximum number of results to return.

    Returns:
        list[dict]: A list of matching movies.
    """
    movies = load_movies()
    search_results = []
    for movie in movies:
        preprocessed_query = preprocess_text(query)
        preprocessed_title = preprocess_text(movie["title"])

        if preprocessed_query in preprocessed_title:
            search_results.append(movie)
            if len(search_results) >= limit:
                break

    return search_results


def preprocess_text(text: str) -> str:
    """Preprocess the input text by lowercasing and removing punctuation.

    Args:
        text (str): The input text.
    Returns:
        str: The preprocessed text.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    return text