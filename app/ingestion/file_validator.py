
from pathlib import Path

# Allowed document file extensions
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".txt",
}

# Maximum allowed upload size: 20 MB
MAX_FILE_SIZE = 20 * 1024 * 1024


def validate_file(filename: str, file_size: int) -> bool:
    """
    Validate the uploaded file's extension and size.

    Args:
        filename: Original uploaded filename.
        file_size: Size of the uploaded file in bytes.

    Returns:
        True if the file passes validation.

    Raises:
        ValueError: If the file type or size is invalid.
    """

    extension = Path(filename).suffix.lower()

    # Check file extension
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    # Check for empty files
    if file_size <= 0:
        raise ValueError(
            "Empty files are not allowed."
        )

    # Check maximum file size
    if file_size > MAX_FILE_SIZE:
        raise ValueError(
            "File exceeds the 20 MB limit."
        )

    return True
