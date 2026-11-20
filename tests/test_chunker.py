
import pytest

from app.ingestion.chunker import chunk_text, chunk_pages


def test_chunk_text_creates_multiple_chunks():

    text = "This is a sample sentence. " * 100

    chunks = chunk_text(
        text,
        chunk_size=200,
        overlap=40,
    )

    assert len(chunks) > 1

    assert all(len(chunk) <= 200 for chunk in chunks)

def test_empty_text_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   ") == []



def test_invalid_chunk_parameters():

    with pytest.raises(ValueError):
        chunk_text("Sample text", chunk_size=100, overlap=100)


def test_chunk_pages_preserves_page_numbers():

    pages = [
        {
            "page_number": 1,
            "text": "Company leave policy. " * 30,
        },
        {
            "page_number": 2,
            "text": "Manager approval policy. " * 30,
        },
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=150,
        overlap=30,
    )

    assert len(chunks) > 2

    assert any(chunk["page_number"] == 1 for chunk in chunks)
    assert any(chunk["page_number"] == 2 for chunk in chunks)

    assert all("text" in chunk for chunk in chunks)
