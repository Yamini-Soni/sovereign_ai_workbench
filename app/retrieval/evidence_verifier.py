
import re


def _split_sentences(text: str) -> list[str]:
    """Split a document chunk into individual sentences."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        if sentence.strip()
    ]


def _find_matching_evidence(
    chunks: list[dict],
    required_keyword_pattern: str,
    value_pattern: str,
) -> dict | None:
    """Find a sentence containing both the required topic and value."""

    keyword_pattern = re.compile(required_keyword_pattern, re.IGNORECASE)
    number_pattern = re.compile(value_pattern, re.IGNORECASE)

    for chunk in chunks:
        for sentence in _split_sentences(chunk["text"]):
            if not keyword_pattern.search(sentence):
                continue

            match = number_pattern.search(sentence)

            if match:
                return {
                    "value": match.group(1),
                    "unit": match.group(2) if match.lastindex >= 2 else None,
                    "evidence": sentence,
                    "filename": chunk.get("filename", "unknown_document"),
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                }

    return None


def verify_policy_facts(
    question: str,
    chunks: list[dict],
) -> dict | None:
    """
    Extract annual leave, remote-work and maternity-leave facts
    directly from retrieved document sentences.

    Returns None when the question does not ask about these facts.
    """

    question_lower = question.lower()

    asks_annual_leave = (
        "annual leave" in question_lower
        or "paid leave" in question_lower
    )

    asks_remote_work = bool(
        re.search(
            r"\bremote\b|\bremotely\b|work from home",
            question_lower,
        )
    )

    asks_maternity = "maternity" in question_lower

    if not any((asks_annual_leave, asks_remote_work, asks_maternity)):
        return None

    answers = []
    evidence_records = []

    if asks_annual_leave:
        evidence = _find_matching_evidence(
            chunks,
            required_keyword_pattern=r"\bannual leave\b",
            value_pattern=r"\b(\d+)\s+(days?)\b",
        )

        if evidence:
            answers.append(
                f"Employees receive {evidence['value']} days "
                f"of annual leave per year. "
                f"[{evidence['filename']} | page {evidence['page_number']}]"
            )
            evidence_records.append({
                "fact": "annual_leave",
                **evidence,
            })
        else:
            answers.append(
                "Annual leave duration could not be found "
                "in the retrieved documents."
            )

    if asks_remote_work:
        evidence = _find_matching_evidence(
            chunks,
            required_keyword_pattern=(
                r"\bremote\b|\bremotely\b|work from home"
            ),
            value_pattern=(
                r"\b(?:up to\s+)?(\d+)\s+days?\s+per\s+week\b"
            ),
        )

        if evidence:
            answers.append(
                f"Employees can work remotely up to "
                f"{evidence['value']} days per week. "
                f"[{evidence['filename']} | page {evidence['page_number']}]"
            )
            evidence_records.append({
                "fact": "remote_work",
                **evidence,
            })
        else:
            answers.append(
                "Remote-work days per week could not be found "
                "in the retrieved documents."
            )

    if asks_maternity:
        evidence = _find_matching_evidence(
            chunks,
            required_keyword_pattern=r"\bmaternity\b",
            value_pattern=(
                r"\b(\d+)\s+(days?|weeks?|months?)\b"
            ),
        )

        if evidence:
            answers.append(
                f"Maternity leave duration is "
                f"{evidence['value']} {evidence['unit']}. "
                f"[{evidence['filename']} | page {evidence['page_number']}]"
            )
            evidence_records.append({
                "fact": "maternity_leave",
                **evidence,
            })
        else:
            answers.append(
                "Maternity leave duration could not be found "
                "in the retrieved documents."
            )

    return {
        "answer": "\n\n".join(
            f"{index}. {answer}"
            for index, answer in enumerate(answers, start=1)
        ),
        "evidence": evidence_records,
        "method": "deterministic_policy_evidence_extraction",
    }


def verify_password_length(
    question: str,
    chunks: list[dict],
) -> dict | None:
    """Extract an explicit minimum password-length requirement."""

    question_lower = question.lower()

    asks_about_password = bool(
        re.search(r"\bpasswords?\b", question_lower)
    )

    asks_about_length = bool(
        re.search(
            r"\blength\b|\bcharacters?\b|\bminimum\b",
            question_lower,
        )
    )

    if not asks_about_password or not asks_about_length:
        return None

    patterns = [
        re.compile(
            r"\bpasswords?\b.{0,100}?"
            r"\b(?:at least|minimum of|no fewer than)\s+"
            r"(\d+)\s+characters?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bpasswords?\b.{0,100}?"
            r"\bminimum\s+(?:length\s+)?(?:of\s+)?"
            r"(\d+)\s+characters?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bpasswords?\b.{0,100}?"
            r"\b(?:contain|include|require|must have)\s+"
            r"(?:at least\s+)?(\d+)\s+characters?\b",
            re.IGNORECASE,
        ),
    ]

    for chunk in chunks:
        for sentence in _split_sentences(chunk["text"]):
            for pattern in patterns:
                match = pattern.search(sentence)

                if match:
                    number = match.group(1)

                    return {
                        "answer": (
                            f"Passwords must contain at least "
                            f"{number} characters."
                        ),
                        "evidence": sentence,
                        "filename": chunk.get("filename"),
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "method": "deterministic_evidence_extraction",
                    }

    return None
