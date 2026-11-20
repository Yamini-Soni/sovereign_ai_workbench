
from pathlib import Path
import fitz


def extract_text(file_path: str) -> dict:
    """
    Extract text from supported document formats.

    Currently supports:
    - PDF
    - TXT

    Returns extracted text and page-level metadata.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError("Document not found.")

    if extension == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")

        return {
            "text": text,
            "pages": [
                {
                    "page_number": 1,
                    "text": text
                }
            ],
            "page_count": 1
        }

    elif extension == ".pdf":
        extracted_pages = []
        full_text = []

        with fitz.open(path) as pdf:

            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text")

                extracted_pages.append({
                    "page_number": page_number,
                    "text": page_text
                })

                full_text.append(page_text)

        return {
            "text": "\n".join(full_text),
            "pages": extracted_pages,
            "page_count": len(extracted_pages)
        }

    else:
        raise ValueError(
            f"Text extraction is not supported for {extension} yet."
        )
