
import json
import sqlite3

from app.core.config import BASE_DIR


DATABASE_PATH = (
    BASE_DIR / "data" / "vector_store.db"
)

DATABASE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_vector_store():
    """
    Create the SQLite table used to store document chunks
    and their embedding vectors.
    """

    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS document_vectors (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_document_vectors_document_id
            ON document_vectors(document_id)
        """)


def store_embeddings(
    document_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
):
    """
    Persist document chunks and their embeddings.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    records = []

    for chunk, embedding in zip(chunks, embeddings):

        if len(embedding) != 384:
            raise ValueError(
                "Expected a 384-dimensional embedding."
            )

        record_id = (
            f"{document_id}_{chunk['chunk_index']}"
        )

        records.append((
            record_id,
            document_id,
            chunk["chunk_index"],
            chunk["page_number"],
            chunk["text"],
            json.dumps(embedding),
        ))

    with get_connection() as connection:
        connection.executemany("""
            INSERT OR REPLACE INTO document_vectors (
                id,
                document_id,
                chunk_index,
                page_number,
                text,
                embedding
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, records)


def get_vector_count() -> int:
    """Return the total number of stored vectors."""

    with get_connection() as connection:
        result = connection.execute("""
            SELECT COUNT(*) AS total
            FROM document_vectors
        """).fetchone()

        return result["total"]


def get_document_vectors(document_id: str) -> list[dict]:
    """Retrieve all stored vectors for one document."""

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM document_vectors
            WHERE document_id = ?
            ORDER BY chunk_index
        """, (document_id,)).fetchall()

    return [
        {
            "id": row["id"],
            "document_id": row["document_id"],
            "chunk_index": row["chunk_index"],
            "page_number": row["page_number"],
            "text": row["text"],
            "embedding": json.loads(row["embedding"]),
        }
        for row in rows
    ]




def get_all_vectors(
    document_id: str | None = None,
) -> list[dict]:
    """
    Retrieve all stored document chunks and embeddings.

    If document_id is provided, retrieve chunks
    only from that particular document.
    """

    with get_connection() as connection:

        if document_id is not None:
            rows = connection.execute("""
                SELECT *
                FROM document_vectors
                WHERE document_id = ?
                ORDER BY chunk_index
            """, (document_id,)).fetchall()

        else:
            rows = connection.execute("""
                SELECT *
                FROM document_vectors
                ORDER BY document_id, chunk_index
            """).fetchall()

    return [
        {
            "id": row["id"],
            "document_id": row["document_id"],
            "chunk_index": row["chunk_index"],
            "page_number": row["page_number"],
            "text": row["text"],
            "embedding": json.loads(row["embedding"]),
        }
        for row in rows
    ]



def delete_document_vectors(document_id: str) -> int:
    """Delete all vectors belonging to one document."""

    with get_connection() as connection:
        cursor = connection.execute(
            """
            DELETE FROM document_vectors
            WHERE document_id = ?
            """,
            (document_id,),
        )

        return cursor.rowcount
