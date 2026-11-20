
from app.retrieval.vector_store import (
    initialize_vector_store,
    get_vector_count,
    get_document_vectors,
)

initialize_vector_store()

print("Total stored vectors:", get_vector_count())

document_id = input("Enter document ID: ").strip()

vectors = get_document_vectors(document_id)

print("Vectors for this document:", len(vectors))

if vectors:
    first = vectors[0]

    print("\nFirst stored chunk:")
    print("Page:", first["page_number"])
    print("Chunk index:", first["chunk_index"])
    print("Text:", first["text"][:200])
    print("Embedding dimensions:", len(first["embedding"]))
