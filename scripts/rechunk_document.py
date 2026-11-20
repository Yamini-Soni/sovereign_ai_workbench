
import json

from app.core.config import BASE_DIR
from app.ingestion.document_extractor import extract_text
from app.ingestion.chunker import chunk_pages

DOCUMENT_ID = "cde1a86f-ea1f-4e87-9132-6aba760a1cf6"

uploaded_file = (
    BASE_DIR
    / "app"
    / "storage"
    / "uploads"
    / f"{DOCUMENT_ID}.txt"
)

if not uploaded_file.exists():
    raise FileNotFoundError(
        f"Uploaded document not found: {uploaded_file}"
    )

# Extract the existing document again.
extraction_result = extract_text(str(uploaded_file))

# Generate smaller chunks.
chunks = chunk_pages(
    extraction_result["pages"],
    chunk_size=250,
    overlap=40,
)

# Save the regenerated chunks.
chunks_file = (
    BASE_DIR
    / "data"
    / "processed"
    / f"{DOCUMENT_ID}_chunks.json"
)

chunks_file.write_text(
    json.dumps(chunks, indent=4, ensure_ascii=False),
    encoding="utf-8",
)

print("Chunk regeneration completed.")
print(f"Document ID: {DOCUMENT_ID}")
print(f"Total chunks: {len(chunks)}")

for chunk in chunks:
    print("\n" + "=" * 60)
    print(f"Chunk: {chunk['chunk_index']}")
    print(f"Page: {chunk['page_number']}")
    print(chunk["text"])
