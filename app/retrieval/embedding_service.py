
from fastembed import TextEmbedding


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 32

# Load the model only once and reuse it.
_embedding_model = None


def get_embedding_model() -> TextEmbedding:
    """
    Load the local embedding model when first required.
    Subsequent calls reuse the same model instance.
    """
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = TextEmbedding(
            model_name=MODEL_NAME
        )

    return _embedding_model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text chunks into embedding vectors.

    Each vector contains 384 floating-point values.
    """

    # Remove empty or whitespace-only inputs.
    cleaned_texts = [
        text.strip()
        for text in texts
        if isinstance(text, str) and text.strip()
    ]

    if not cleaned_texts:
        return []

    model = get_embedding_model()

    embedding_generator = model.embed(
        cleaned_texts,
        batch_size=BATCH_SIZE,
    )

    embeddings = [
        embedding.tolist()
        for embedding in embedding_generator
    ]

    # Verify the model output.
    for embedding in embeddings:
        if len(embedding) != EMBEDDING_DIMENSION:
            raise RuntimeError(
                "Unexpected embedding dimension: "
                f"{len(embedding)}"
            )

    return embeddings
