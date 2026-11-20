
import json
from pathlib import Path

from app.retrieval.embedding_service import (
    generate_embeddings,
)
from app.retrieval.vector_store import (
    initialize_vector_store,
    store_embeddings,
    get_vector_count,
)


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def main():

    initialize_vector_store()

    chunk_files = list(
        PROCESSED_DIR.glob("*_chunks.json")
    )

    if not chunk_files:
        print("No chunk files found.")
        return

    total_processed = 0

    for chunk_file in chunk_files:

        document_id = chunk_file.stem.replace(
            "_chunks", ""
        )

        print(f"\nProcessing document: {document_id}")

        with chunk_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            chunks = json.load(file)

        if not chunks:
            print("No chunks available. Skipping.")
            continue

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print(f"Generating embeddings for {len(texts)} chunks...")

        embeddings = generate_embeddings(texts)

        store_embeddings(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
        )

        total_processed += len(embeddings)

        print(
            f"Stored {len(embeddings)} embeddings."
        )

    print("\nVector storage completed.")
    print(f"Vectors processed: {total_processed}")
    print(f"Total vectors in database: {get_vector_count()}")


if __name__ == "__main__":
    main()
