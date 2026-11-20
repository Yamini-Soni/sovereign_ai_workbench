
"""Registry of approved, read-only tools available to the agent."""

import ast
import math
import operator
from app.vision.vision_service import analyze_image
from app.retrieval.answer_service import answer_question
from app.retrieval.hybrid_search import hybrid_search
from app.storage.metadata_store import get_all_documents


# ---------------------------------------------------------
# Tool 1: Existing RAG question-answering pipeline
# ---------------------------------------------------------

def _answer_document_question(question: str) -> dict:
    """Answer a question using the existing grounded RAG pipeline."""

    result = answer_question(question=question)

    # Preserve the answer and its source information.
    return {
        "answer": result.get("answer"),
        "answer_method": result.get("answer_method"),
        "verified_evidence": result.get("verified_evidence"),
        "sources": result.get("sources", []),
    }


# ---------------------------------------------------------
# Tool 2: Hybrid document search
# ---------------------------------------------------------

# ---------------------------------------------------------
# Tool 5: Local image understanding
# ---------------------------------------------------------

def _analyze_image(question: str, image_bytes: bytes) -> dict:
    """Answer a question using the validated request image."""

    result = analyze_image(
        image_bytes=image_bytes,
        task="vqa",
        question=question,
    )

    return {
        "model": result["model"],
        "task": result["task"],
        "description": result["description"],
    }

def _search_documents(query: str, top_k: int = 5) -> dict:
    """Search approved documents using semantic + lexical retrieval."""

    results = hybrid_search(
        query=query,
        top_k=top_k,
    )

    # Build a document ID -> original filename mapping.
    documents = get_all_documents()

    document_lookup = {
        document["document_id"]: document.get(
            "filename",
            "Unknown document",
        )
        for document in documents
        if document.get("document_id")
    }

    matches = []

    for result in results:
        document_id = result["document_id"]

        matches.append({
            "document_id": document_id,
            "filename": document_lookup.get(
                document_id,
                f"document_{document_id}",
            ),
            "page_number": result["page_number"],
            "chunk_index": result["chunk_index"],
            "snippet": result["text"][:1000],
            "rrf_score": result["rrf_score"],
            "semantic_score": result["semantic_score"],
            "bm25_score": result["bm25_score"],
        })

    return {
        "query": query,
        "match_count": len(matches),
        "matches": matches,
    }


# ---------------------------------------------------------
# Tool 3: List uploaded documents
# ---------------------------------------------------------

def _list_documents() -> dict:
    """Return basic metadata for uploaded documents."""

    documents = get_all_documents()

    safe_documents = []

    for document in documents:
        safe_documents.append({
            "document_id": document.get("document_id"),
            "filename": document.get("filename"),
            "extension": document.get("extension"),
            "file_size_bytes": document.get("file_size_bytes"),
            "page_count": document.get("page_count"),
            "extraction_status": document.get(
                "extraction_status"
            ),
            "chunk_count": document.get("chunk_count"),
        })

    return {
        "document_count": len(safe_documents),
        "documents": safe_documents,
    }


# ---------------------------------------------------------
# Tool 4: Restricted mathematical calculator
# ---------------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

MAX_EXPRESSION_LENGTH = 100
MAX_AST_NODES = 30
MAX_ABSOLUTE_RESULT = 1e100


def _evaluate_math_node(node):
    """Evaluate only explicitly permitted arithmetic expressions."""

    if isinstance(node, ast.Expression):
        return _evaluate_math_node(node.body)

    if isinstance(node, ast.Constant):
        # Do not accept booleans, strings, or other constants.
        if isinstance(node.value, bool):
            raise ValueError("Boolean values are not allowed.")

        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric values are allowed.")

        return node.value

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("This mathematical operator is not allowed.")

        left = _evaluate_math_node(node.left)
        right = _evaluate_math_node(node.right)

        result = _ALLOWED_OPERATORS[operator_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("This unary operator is not allowed.")

        operand = _evaluate_math_node(node.operand)
        result = _ALLOWED_OPERATORS[operator_type](operand)

    else:
        raise ValueError("This expression contains an unsupported operation.")

    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError("The expression must produce a number.")

    if not math.isfinite(result):
        raise ValueError("The result must be finite.")

    if abs(result) > MAX_ABSOLUTE_RESULT:
        raise ValueError("The result is too large.")

    return result


def _calculate(expression: str) -> dict:
    """Safely evaluate a restricted arithmetic expression."""

    expression = expression.strip()

    if not expression:
        raise ValueError("Expression cannot be empty.")

    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long.")

    # Parse into an AST; never use eval() or exec().
    tree = ast.parse(expression, mode="eval")

    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Expression is too complex.")

    result = _evaluate_math_node(tree)

    return {
        "expression": expression,
        "result": result,
    }


# ---------------------------------------------------------
# Tool registry
# ---------------------------------------------------------

TOOLS = {
    "answer_document_question": {
        "description": (
            "Answer a question using the approved documents "
            "and the existing grounded RAG pipeline. "
            "Use this when the user asks for an answer from documents."
        ),
        "parameters": {
            "question": {
                "type": "string",
                "required": True,
                "max_length": 2000,
            },
        },
        "function": _answer_document_question,
    },

    "search_documents": {
        "description": (
            "Search approved documents for relevant evidence "
            "using hybrid semantic and lexical retrieval. "
            "Use this to find relevant passages."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "required": True,
                "max_length": 1000,
            },
            "top_k": {
                "type": "integer",
                "required": False,
                "default": 5,
                "min": 1,
                "max": 5,
            },
        },
        "function": _search_documents,
    },

    "list_documents": {
        "description": (
            "List uploaded approved documents and their basic metadata. "
            "Use this when the user asks which documents are available."
        ),
        "parameters": {},
        "function": _list_documents,
    },

    "calculate": {
        "description": (
            "Calculate a basic mathematical expression using "
            "addition, subtraction, multiplication, division, "
            "modulo, and unary plus or minus."
        ),
        "parameters": {
            "expression": {
                "type": "string",
                "required": True,
                "max_length": MAX_EXPRESSION_LENGTH,
            },
        },
        "function": _calculate,
    },
        "analyze_image": {
        "description": (
            "Analyze the image attached to the current user request "
            "and answer a question using visible visual evidence. "
            "Use this for image understanding and visual question answering."
        ),
        "parameters": {
            "question": {
                "type": "string",
                "required": True,
                "max_length": 2000,
            },
        },
        "function": _analyze_image,
    },
}


def describe_tools() -> list[dict]:
    """Return tool descriptions for the local LLM planner."""

    return [
        {
            "name": name,
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for name, tool in TOOLS.items()
    ]
