import os
import re
import json
import numpy as np

from sentence_transformers import SentenceTransformer

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DOCUMENT_PREVIEW_LENGTH,
    SCORE_PRECISION,
    MODEL_NAME,
    MOVIE_EMBEDDINGS_PATH,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    format_search_result,
    load_movies,
)


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
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunk(
                text,
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
            )

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": idx, "chunk_idx": i, "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Load chunk embeddings from cache or create them if not available."""
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            if len(self.chunk_embeddings) == len(self.chunk_metadata):
                return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        """Search chunked documents using cosine similarity."""
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )

        query_embedding = self.generate_embedding(query)
        chunk_scores = []

        for index, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": index,
                    "movie_idx": self.chunk_metadata[index]["movie_idx"],
                    "score": similarity_score,
                }
            )

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score

        ranked_movies = sorted(
            movie_scores.items(), key=lambda item: item[1], reverse=True
        )

        formatted_results = []
        for movie_idx, score in ranked_movies[:limit]:
            if movie_idx is None:
                continue
            doc = self.documents[movie_idx]
            formatted_results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )

        return formatted_results


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


def embed_chunks() -> None:
    """Generate and print embeddings for chunks of the given text."""
    documents = load_movies()
    search_instance = ChunkedSemanticSearch()
    embeddings = search_instance.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_search(query, limit=DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    search_instance = SemanticSearch()
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
) -> list[str]:
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

    return chunks


def chunk_print_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    """Chunk the given text into smaller pieces and print them."""
    chunks = chunk(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk_text in enumerate(chunks):
        print(f"{i + 1}. {chunk_text}")


def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
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

    return chunks


def semantic_chunk_print_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    """Chunk the given text semantically into smaller pieces and print them."""
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk_text in enumerate(chunks):
        print(f"{i + 1}. {chunk_text}")


def search_chunked(query: str, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    search_instance = ChunkedSemanticSearch()
    search_instance.load_or_create_chunk_embeddings(movies)
    results = search_instance.search_chunks(query, limit)

    for i, result in enumerate(results, 1):
        title = result["title"]
        score = result["score"]
        description = result["document"]
        print(f"{i}. {title} (score: {score:.4f})")
        print(f"   {description}...")
