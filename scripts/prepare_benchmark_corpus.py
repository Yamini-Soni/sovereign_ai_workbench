"""Index the versioned synthetic benchmark corpus under stable document IDs."""

from pathlib import Path

from app.core.config import BASE_DIR
from app.ingestion.chunker import chunk_pages
from app.retrieval.embedding_service import generate_embeddings
from app.retrieval.vector_store import initialize_vector_store, store_embeddings


CORPUS_DIR = BASE_DIR / "benchmarks" / "synthetic_industrial_corpus"
BENCHMARK_DOCUMENTS = (
    "nexatech_company_policy.txt",
    "industrial_maintenance_manual.txt",
)


def prepare_benchmark_corpus() -> dict:
    """Embed benchmark files using their filenames as stable evaluation IDs."""
    initialize_vector_store()
    indexed = {}
    for filename in BENCHMARK_DOCUMENTS:
        text = (CORPUS_DIR / filename).read_text(encoding="utf-8")
        chunks = chunk_pages([{"page_number": 1, "text": text}], chunk_size=250, overlap=40)
        embeddings = generate_embeddings([chunk["text"] for chunk in chunks])
        store_embeddings(filename, chunks, embeddings)
        indexed[filename] = len(chunks)
    return {"documents": indexed, "chunks": sum(indexed.values())}


if __name__ == "__main__":
    print(prepare_benchmark_corpus())
