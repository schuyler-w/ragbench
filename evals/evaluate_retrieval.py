import json, numpy as np
from tqdm import tqdm
from src.rag import RAGPipeline

EVAL_PATH   = "evals/eval_set.jsonl"
RESULTS_OUT = "results/retrieval_metrics.json"
KS          = [1, 3, 5, 10]

def recall_at_k(retrieved_ids: list, relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0

def mrr_score(retrieved_ids: list, relevant_id: str) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid == relevant_id:
            return 1.0 / (i + 1)
    return 0.0

def evaluate():
    import os
    os.makedirs("results", exist_ok=True)

    rag = RAGPipeline()
    examples = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8")]

    scores = {f"recall@{k}": [] for k in KS}
    scores["mrr"] = []

    for ex in tqdm(examples, desc="Evaluating retrieval"):
        retrieved = rag.retrieve(ex["question"], k=max(KS))
        retrieved_ids = [m["chunk_id"] for _, m, _ in retrieved]

        for k in KS:
            scores[f"recall@{k}"].append(
                recall_at_k(retrieved_ids, ex["relevant_chunk_id"], k)
            )
        scores["mrr"].append(
            mrr_score(retrieved_ids, ex["relevant_chunk_id"])
        )

    results = {metric: round(float(np.mean(vals)), 4)
               for metric, vals in scores.items()}

    print("\n=== Retrieval Evaluation Results ===")
    for metric, val in results.items():
        print(f"  {metric:12s}: {val:.4f}")

    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_OUT}")
    return results

if __name__ == "__main__":
    evaluate()