
import json
from pathlib import Path

from app.retrieval.embedding_service import (
    generate_embeddings,
    MODEL_NAME,
    EMBEDDING_DIMENSION,
)


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def main():
    chunk_files = list(
        PROCESSED_DIR.glob("*_chunks.json")
    )

    if not chunk_files:
        print("No chunk files found.")
        print("Upload a PDF or TXT document first.")
        return

    # Select the most recently modified chunk file.
    latest_file = max(
        chunk_files,
        key=lambda path: path.stat().st_mtime,
    )

    print(f"Using document: {latest_file.name}")

    with latest_file.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        print("The selected document contains no chunks.")
        return

    # Embed up to three chunks for the initial test.
    sample_chunks = chunks[:3]

    texts = [
        chunk["text"]
        for chunk in sample_chunks
    ]

    print(f"Loading model: {MODEL_NAME}")
    print("Generating embeddings...")

    embeddings = generate_embeddings(texts)

    print("\nEmbedding test successful.")
    print(f"Chunks tested: {len(texts)}")
    print(f"Embeddings generated: {len(embeddings)}")
    print(f"Expected vector dimension: {EMBEDDING_DIMENSION}")

    for index, embedding in enumerate(embeddings):
        print(
            f"Chunk {index}: vector dimension = "
            f"{len(embedding)}"
        )

    print("\nFirst five values of the first vector:")
    print(embeddings[0][:5])


if __name__ == "__main__":
    main()
