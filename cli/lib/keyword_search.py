import string

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


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
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            search_results.append(movie)
            if len(search_results) >= limit:
                break

    return search_results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    """Check if there is any matching token between query and title tokens.

    Args:
        query_tokens (list[str]): List of tokens from the search query.
        title_tokens (list[str]): List of tokens from the movie title.

    Returns:
        bool: True if there is at least one matching token, False otherwise.
    """
    for query_token in query_tokens:
        for title_token in title_tokens:
            if title_token.find(query_token) != -1:
                return True
            
    return False


def preprocess_text(text: str) -> list[str]:
    """Preprocess the input text by lowercasing and removing punctuation.

    Args:
        text (str): The input text.
    Returns:
        str: The preprocessed text.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords from the list of tokens.

    Args:
        tokens (list[str]): The list of tokens.

    Returns:
        list[str]: A list of tokens with stopwords removed.
    """
    stopwords = load_stopwords()
    return [token for token in tokens if token not in stopwords]


def tokenize_text(text: str) -> list[str]:
    """Tokenize the input text into words.

    Args:
        text (str): The input text.
    Returns:
        list[str]: A list of tokens.
    """
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    valid_tokens = remove_stopwords(valid_tokens)

    return valid_tokens