
# Baseline Experiment 01 — Local RAG

## Configuration
- LLM: llama3.2:3b
- Inference: Ollama (local)
- Embedding model: all-MiniLM-L6-v2
- Vector store: SQLite
- Retrieval: Semantic search
- Top K: 3
- Minimum similarity: 0.30
- Prompt: v2-detailed
- Dataset: NexaTech Company Policy
- Evaluation questions: 15

## Answer Evaluation
- Answerable questions: 9/10 (90%)
- Unanswerable questions correctly refused: 5/5 (100%)
- Overall keyword-based pass rate: 14/15 (93.3%)
- Median end-to-end response time: 2.67 seconds
- Failed question: Q05 (minimum password length)

## Hardware
- GPU: NVIDIA RTX 3050 Laptop GPU
- VRAM capacity: 6144 MiB
- Observed GPU memory usage: 2539 MiB
- Observed peak GPU utilization: 99%
- Observed maximum temperature: 51°C
- Observed peak power: 36.19 W

## Observation
The model successfully answers most questions and refuses
unsupported questions. Q05 reveals a generation-level
contradiction despite successful retrieval of the correct evidence.

## Limitations
- Only one document used.
- Only 15 evaluation questions.
- Keyword-based evaluation is preliminary.
- GPU measurements are sampled observations.
- Results cannot yet establish general RAG performance.
