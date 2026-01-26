import os
import pickle
import string
import math
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_K1,
    BM25_B,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    """Inverted index for keyword search."""

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self) -> None:
        """Build the inverted index from the movie dataset."""
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        """Save the inverted index, docmap, term frequencies, and document lengths to disk."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        """Load the inverted index, docmap, term frequencies, and document lengths from disk."""
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError:
            raise Exception(
                "Inverted index, docmap, term_frequencies, or doc_lengths not found. Please build the index first."
            )

    def get_documents(self, term: str) -> list[int]:
        """Get document IDs containing the given term."""
        doc_ids = self.index.get(term.lower(), set())
        return sorted(list(doc_ids)) if doc_ids else []

    def get_tf(self, doc_id: int, term: str) -> int:
        """Get term frequency for a document and term."""
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token.")
        term = tokens[0]
        return self.term_frequencies.get(doc_id, {}).get(term, 0)

    def get_idf(self, term: str) -> float:
        """Get inverse document frequency for a given term."""
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token.")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        """Get TF-IDF score for a document and term."""
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        """Get BM25 inverse document frequency for a given term."""
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token.")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        """Get BM25 term frequency for a given document ID and term."""
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length == 0:
            return 0.0
        norm_factor = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * norm_factor)

    def bm25(self, doc_id: int, term: str) -> float:
        """Calculate BM25 score for a given document ID and term."""
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit=DEFAULT_SEARCH_LIMIT) -> list[dict]:
        """Search movies using BM25 ranking."""
        query_tokens = tokenize_text(query)
        scores = defaultdict(float)
        for token in query_tokens:
            for doc_id in self.get_documents(token):
                scores[doc_id] += self.bm25(doc_id, token)
        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_docs = ranked_docs[:limit]
        results = []
        for doc_id, score in top_docs:
            doc = self.docmap[doc_id].copy()
            doc["score"] = score
            results.append(doc)
        return results

    def __add_document(self, doc_id: int, text: str) -> None:
        """Add a document to the inverted index."""
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        """Calculate the average document length."""
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths) if self.doc_lengths else 0.0


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


def idf_command(term: str) -> float:
    """Get inverse document frequency for a term.

    Args:
        term (str): Term to check.

    Returns:
        float: Inverse document frequency of the term.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    """Get TF-IDF score for a document and term.

    Args:
        doc_id (int): Document ID.
        term (str): Term to check.

    Returns:
        float: TF-IDF score of the term in the document.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)


def bm25_idf_command(term: str) -> float:
    """Get BM25 inverse document frequency for a term.

    Args:
        term (str): Term to check.

    Returns:
        float: BM25 inverse document frequency of the term.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
    """Get BM25 term frequency for a document and term.

    Args:
        doc_id (int): Document ID.
        term (str): Term to check.
        k1 (float): BM25 k1 parameter.
        b (float): BM25 b parameter.

    Returns:
        float: BM25 term frequency of the term in the document.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)


def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """Search movies using BM25 ranking.

    Args:
        query (str): The search query.
        limit (int): The maximum number of results to return.

    Returns:
        list[dict]: A list of matching movies ranked by BM25 score.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)


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
