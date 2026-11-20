from pathlib import Path
from app.ingestion.document_extractor import extract_text


def test_txt_extraction(tmp_path):

    sample_file = tmp_path / "sample.txt"

    sample_file.write_text(
        "Sovereign AI Workbench test document.",
        encoding="utf-8"
    )

    result = extract_text(str(sample_file))

    assert result["page_count"] == 1

    assert (
        "Sovereign AI Workbench"
        in result["text"]
    )


def test_missing_file():

    try:
        extract_text("nonexistent.pdf")
        assert False, "Expected FileNotFoundError"

    except FileNotFoundError:
        assert True