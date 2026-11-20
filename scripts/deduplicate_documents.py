
import hashlib
import json
import sqlite3

from pathlib import Path

from app.core.config import BASE_DIR


# Keep the latest document.
KEEP_ID = "417b52cf-4e7f-4faf-a616-a3b7603923fd"

# Remove the two older duplicate records.
REMOVE_IDS = [
    "cde1a86f-ea1f-4e87-9132-6aba760a1cf6",
    "db92d3dd-8c1a-4462-a8a4-9d9c7eb0f58c",
]

# Correct SHA-256 hash verified from all three files.
EXPECTED_HASH = (
    "3ec25f28643243ab755304c82e6bb4fa2a13fd43bf6cf2f6f2cfc8a4aa647e0c"
)

METADATA_FILE = (
    BASE_DIR / "app" / "storage" / "documents_metadata.json"
)

UPLOAD_DIR = BASE_DIR / "app" / "storage" / "uploads"

DATABASE_PATH = BASE_DIR / "data" / "vector_store.db"


def calculate_hash(path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main():

    print("=" * 65)
    print("DUPLICATE DOCUMENT CLEANUP")
    print("=" * 65)

    # 1. Load metadata.
    with METADATA_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    required_ids = [KEEP_ID] + REMOVE_IDS

    matching_documents = {
        document["document_id"]: document
        for document in documents
        if document["document_id"] in required_ids
    }

    if len(matching_documents) != 3:
        raise RuntimeError(
            "Expected all three document records. Cleanup cancelled."
        )

    # 2. Verify all three uploaded files.
    print("\nVerifying uploaded files...")

    for document_id in required_ids:

        document = matching_documents[document_id]

        file_path = UPLOAD_DIR / document["stored_filename"]

        if not file_path.exists():
            raise RuntimeError(
                f"Uploaded file is missing: {file_path}"
            )

        actual_hash = calculate_hash(file_path)

        if actual_hash != EXPECTED_HASH:
            raise RuntimeError(
                f"Hash mismatch for {document_id}. Cleanup cancelled."
            )

        print(f"Verified: {document_id}")

    # 3. Connect to vector database.
    connection = sqlite3.connect(DATABASE_PATH)

    try:

        # Check vector counts before deleting anything.
        counts = {}

        for document_id in required_ids:

            result = connection.execute(
                """
                SELECT COUNT(*)
                FROM document_vectors
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

            counts[document_id] = result[0]

        print("\nVector counts before cleanup:")

        for document_id, count in counts.items():
            print(f"{document_id}: {count}")

        if counts[KEEP_ID] != 3:
            raise RuntimeError(
                "Canonical document must have exactly 3 vectors."
            )

        if any(counts[item] != 3 for item in REMOVE_IDS):
            raise RuntimeError(
                "Unexpected duplicate vector count. Cleanup cancelled."
            )

        # 4. Prepare updated metadata.
        canonical = matching_documents[KEEP_ID]

        canonical["content_hash"] = EXPECTED_HASH
        canonical["chunk_count"] = counts[KEEP_ID]

        updated_documents = [
            document
            for document in documents
            if document["document_id"] not in REMOVE_IDS
        ]

        # Write temporary metadata first.
        temporary_file = METADATA_FILE.with_suffix(".tmp")

        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(
                updated_documents,
                file,
                indent=4,
                ensure_ascii=False,
            )

        # 5. Delete duplicate vectors in one transaction.
        with connection:

            for document_id in REMOVE_IDS:

                connection.execute(
                    """
                    DELETE FROM document_vectors
                    WHERE document_id = ?
                    """,
                    (document_id,),
                )

        # 6. Replace metadata after successful database cleanup.
        temporary_file.replace(METADATA_FILE)

    finally:
        connection.close()

    print("\n" + "=" * 65)
    print("CLEANUP COMPLETED")
    print("=" * 65)

    print("Kept document:", KEEP_ID)
    print("Removed metadata records:", len(REMOVE_IDS))
    print("Remaining metadata records:", len(updated_documents))
    print("Expected remaining vectors: 3")
    print("Uploaded files were preserved.")


if __name__ == "__main__":
    main()
