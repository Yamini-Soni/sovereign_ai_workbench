
def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    """
    Divide text into overlapping chunks.

    Args:
        text: Input document text.
        chunk_size: Maximum target characters per chunk.
        overlap: Number of overlapping characters.

    Returns:
        A list of text chunks.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        hard_end = min(start + chunk_size, text_length)
        end = hard_end

        if hard_end < text_length:
            # Prefer splitting at a whitespace boundary.
            minimum_boundary = start + int(chunk_size * 0.7)

            space_position = text.rfind(
                " ",
                minimum_boundary,
                hard_end
            )

            newline_position = text.rfind(
                "\n",
                minimum_boundary,
                hard_end
            )

            boundary = max(space_position, newline_position)

            if boundary != -1:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Move forward while retaining the requested overlap.

        # Preserve overlap while avoiding starting in the middle of a word.
        next_start = max(start + 1, end - overlap)

        if next_start < end:
            remaining_text = text[next_start:end]

            for index, character in enumerate(remaining_text):
                if character.isspace():
                    next_start += index + 1
                    break

        if next_start <= start:
            next_start = end

        start = next_start




        

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """
    Chunk extracted document pages while preserving page numbers.
    """

    all_chunks = []

    for page in pages:

        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for page_chunk_index, chunk in enumerate(page_chunks):

            all_chunks.append({
                "chunk_index": len(all_chunks),
                "page_chunk_index": page_chunk_index,
                "page_number": page_number,
                "text": chunk,
            })

    return all_chunks
