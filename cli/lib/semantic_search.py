import os
import numpy as np

from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies

MODEL_NAME = "all-MiniLM-L6-v2"
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SemanticSearch:
    """Class for semantic search using sentence transformers."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for the given text."""
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[dict]) -> None:
        """Build embeddings for a list of documents."""
        self.documents = documents
        self.document_map = {}
        movies = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            movies.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movies, show_progress_bar=True)

        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Load embeddings from cache or create them if not available."""
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)


def verify_model() -> None:
    """Verify that the sentence transformer model loads correctly."""
    try:
        search_instance = SemanticSearch()
        print(f"Model loaded: {search_instance.model}")
        print(f"Max sequence length: {search_instance.model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")


def verify_embeddings() -> None:
    """Verify that embeddings can be generated."""
    try:
        search_instance = SemanticSearch()
        movies = load_movies()
        search_instance.load_or_create_embeddings(movies)
        print(f"Number of docs:   {len(movies)}")
        print(
            f"Embeddings shape: {search_instance.embeddings.shape[0]} vectors in {search_instance.embeddings.shape[1]} dimensions"
        )
    except Exception as e:
        print(f"Error generating embeddings: {e}")


def embed_text(text: str) -> None:
    """Generate and print the embedding for the given text."""
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    """Generate and return the embedding for the given query text."""
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
