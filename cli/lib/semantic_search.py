from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticSearch:
    """Class for semantic search using sentence transformers."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)


def verify_model() -> None:
    """Verify that the sentence transformer model loads correctly."""
    try:
        search_instance = SemanticSearch()
        print(f"Model loaded: {search_instance.model}")
        print(f"Max sequence length: {search_instance.model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")
