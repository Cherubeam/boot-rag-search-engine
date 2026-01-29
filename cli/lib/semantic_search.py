import os
import re
import numpy as np

from sentence_transformers import SentenceTransformer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    load_movies,
)

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
        movies = list[str]
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

    def search(self, query: str, limit) -> None:
        """Search documents using cosine similarity."""
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for index, doc_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity_score, self.documents[index]))

        similarities.sort(key=lambda x: x[0], reverse=True)
        results = [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in similarities
        ]

        return results[:limit]


class ChunkedSemanticSearch(SemanticSearch):
    """Class for semantic search on chunked documents."""

    def __init__(self, model_name=MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> None:
        """Build embeddings for chunked documents."""
        self.documents = documents
        self.document_map = {}
        chunks = list[str]


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


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_search(query, limit=DEFAULT_SEARCH_LIMIT):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()


def chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    """Chunk the given text into smaller pieces."""
    words = text.split()
    chunks = []

    if overlap > 0:
        i = 0
        while i < len(words):
            if i == 0:
                chunk = words[i : i + chunk_size]
                chunks.append(" ".join(chunk))
                i += chunk_size
            else:
                chunk = words[i - overlap : i - overlap + chunk_size]
                chunks.append(" ".join(chunk))
                i += chunk_size - overlap
    else:
        for i in range(0, len(words), chunk_size):
            chunk = words[i : i + chunk_size]
            chunks.append(" ".join(chunk))

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def semantic_chunk(
    text: str,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    """Chunk the given text semantically into smaller pieces."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    if overlap > 0:
        i = 0
        while i < len(sentences):
            if i == 0:
                chunk = sentences[i : i + max_chunk_size]
                chunks.append(" ".join(chunk))
                i += max_chunk_size
            else:
                chunk = sentences[i - overlap : i - overlap + max_chunk_size]
                chunks.append(" ".join(chunk))
                i += max_chunk_size - overlap
    else:
        for i in range(0, len(sentences), max_chunk_size):
            chunk = sentences[i : i + max_chunk_size]
            chunks.append(" ".join(chunk))

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")
