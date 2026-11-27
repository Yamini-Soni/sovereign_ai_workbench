
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


_CLAIM_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "to", "of", "in", "on", "at", "by", "for", "from",
    "with", "that", "this", "these", "those", "it", "its", "as", "must",
    "may", "can", "should", "will", "would", "could", "has", "have", "had",
}


def _claim_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {
        token
        for token in tokens
        if token not in _CLAIM_STOPWORDS and (len(token) > 2 or token.isdigit())
    }


def verify_answer_claims(answer: str, chunks: list[dict]) -> dict:
    """
    Evaluate whether sentences in a generated answer are lexically and numerically
    grounded in the retrieved source chunks.
    """
    if not answer or not answer.strip():
        return {
            "supported": False,
            "total_sentences": 0,
            "supported_sentences": 0,
            "unsupported_sentences": [],
            "support_ratio": 0.0,
        }

    # Refusal responses do not make document factual claims.
    if "could not find" in answer.lower():
        return {
            "supported": True,
            "total_sentences": 0,
            "supported_sentences": 0,
            "unsupported_sentences": [],
            "support_ratio": 1.0,
        }

    # Strip trailing bracketed citations before inspecting claim text.
    cleaned = re.sub(r"\[[^\]]+\|\s*page\s+\d+\]", "", answer, flags=re.IGNORECASE)
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        if s.strip() and not s.strip().startswith("#")
    ]

    factual_sentences = [
        s for s in sentences if len(_claim_tokens(s)) >= 2
    ]

    if not factual_sentences:
        return {
            "supported": True,
            "total_sentences": 0,
            "supported_sentences": 0,
            "unsupported_sentences": [],
            "support_ratio": 1.0,
        }

    unsupported = []
    supported_count = 0

    for sentence in factual_sentences:
        sent_tokens = _claim_tokens(sentence)
        sent_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", sentence))

        matched = False
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            chunk_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", chunk_text))
            if not sent_numbers.issubset(chunk_numbers):
                continue

            chunk_tokens = _claim_tokens(chunk_text)
            overlap = sent_tokens.intersection(chunk_tokens)
            if len(overlap) >= 2 and (len(overlap) / len(sent_tokens)) >= 0.30:
                matched = True
                break

        if matched:
            supported_count += 1
        else:
            unsupported.append(sentence)

    total = len(factual_sentences)
    ratio = round(supported_count / total, 4) if total else 1.0

    return {
        "supported": len(unsupported) == 0,
        "total_sentences": total,
        "supported_sentences": supported_count,
        "unsupported_sentences": unsupported,
        "support_ratio": ratio,
    }
