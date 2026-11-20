
from app.retrieval.semantic_search import semantic_search
from app.retrieval.evidence_verifier import verify_password_length

def main():

    questions = [
        "What is the minimum password length?",
        "How many characters must a password contain at minimum?",
        "How often must passwords be changed?",
        "What is the CEO's salary?",
    ]

    print("=" * 65)
    print("DETERMINISTIC EVIDENCE VERIFIER TEST")
    print("=" * 65)

    for question in questions:

        print(f"\nQUESTION: {question}")

        chunks = semantic_search(
            query=question,
            top_k=3,
            min_similarity=0.30,
            document_id=None,
        )

        result = verify_password_length(
            question=question,
            chunks=chunks,
        )

        if result:
            print("\nVERIFIED ANSWER:", result["answer"])
            print("EVIDENCE:", result["evidence"])
            print("PAGE:", result["page_number"])
            print("METHOD:", result["method"])
        else:
            print("\nNO VERIFIED ANSWER")
        
        print("-" * 65)

if __name__ == "__main__":
    main()