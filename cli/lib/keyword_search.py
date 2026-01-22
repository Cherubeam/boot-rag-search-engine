import os
import pickle
import string
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies, 
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)
    
    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            raise Exception("Inverted index, docmap, or term_frequencies not found. Please build the index first.")
        
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower(), set())
        return sorted(list(doc_ids)) if doc_ids else []
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token.")
        term = tokens[0]
        return self.term_frequencies.get(doc_id, {}).get(term, 0)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)


def build_command() -> None:
    """Build and save the inverted index."""
    idx = InvertedIndex()
    idx.build()
    idx.save()
    

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """Search movies by title using a simple keyword match.

    Args:
        query (str): The search query.
        limit (int): The maximum number of results to return.

    Returns:
        list[dict]: A list of matching movies.
    """
    idx = InvertedIndex()
    idx.load()
    for token in tokenize_text(query):
        doc_ids = idx.get_documents(token)
        if doc_ids:
            return [idx.docmap[doc_id] for doc_id in doc_ids[:limit]]
        

def tf_command(doc_id: int, term: str) -> int:
    """Get term frequency for a document and term.

    Args:
        doc_id (int): Document ID.
        term (str): Term to check.

    Returns:
        int: Term frequency in the document.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


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


def stem_words(token: str) -> str:
    """Stem the input token using Porter Stemmer.

    Args:
        token (str): The input token.
    Returns:
        str: The stemmed token.
    """
    stemmer = PorterStemmer()
    return stemmer.stem(token)


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
    valid_tokens = [stem_words(token) for token in valid_tokens]

    return valid_tokens