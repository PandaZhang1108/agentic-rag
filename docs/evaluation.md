# Retrieval evaluation

The repository includes a fixed development set covering normal, ambiguous, multi-source, and out-of-scope questions. Retrieval experiments use the same FastAPI documentation corpus and the same 25 answerable questions. Out-of-scope questions are reserved for full-agent evaluation.

## Results

`RETRIEVE_K=5`:

| Strategy | Recall@5 | MRR@5 | p50 latency |
|---|---:|---:|---:|
| Dense | 0.7533 | 0.6713 | 121.7 ms |
| BM25 | **0.8267** | **0.8400** | 50.0 ms |
| Dense + BM25 + RRF | 0.8067 | 0.7833 | 154.7 ms |

BM25 performed best on this Chinese FastAPI documentation set. This result does not establish that BM25 is generally better: hybrid retrieval still helped some individual questions, while other questions regressed. A production choice should be based on the target corpus, per-question failures, and latency constraints.

## Reproduce

```bash
python -m core.evals run \
  --dataset domain/evalset.jsonl \
  --mode retrieval \
  --tag local-retrieval-run
```

Recall and MRR measure retrieval only. They do not prove final-answer correctness. Full-agent evaluation should separately measure correctness, completeness, faithfulness, citations, refusal behavior, tool choice, latency, token use, and cost.
