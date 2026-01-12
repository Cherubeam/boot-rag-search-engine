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
        if movie["title"].lower().find(query.lower()) != -1:
            search_results.append(movie)
            if len(search_results) >= limit:
                break
    return search_results