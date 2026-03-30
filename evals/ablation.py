import json, os, numpy as np
from src.chunk import chunk_corpus
from src.embed import embed_corpus
from src.index import build_index
from src.rag import RAGPipeline

EVAL_PATH  = "evals/eval_set.jsonl"
RESULTS_OUT = "results/ablation_results.json"
KS         = [1, 3, 5, 10]

CONFIGS = [
    {"name": "baseline",     "chunk_size": 256, "overlap": 32, "model": "BAAI/bge-large-en-v1.5"},
    {"name": "small_chunks", "chunk_size": 128, "overlap": 16, "model": "BAAI/bge-large-en-v1.5"},
    {"name": "large_chunks", "chunk_size": 512, "overlap": 64, "model": "BAAI/bge-large-en-v1.5"},
    {"name": "no_overlap",   "chunk_size": 256, "overlap": 0,  "model": "BAAI/bge-large-en-v1.5"},
    {"name": "gte_large",    "chunk_size": 256, "overlap": 32, "model": "thenlper/gte-large"},
    {"name": "mxbai",        "chunk_size": 256, "overlap": 32, "model": "mixedbread-ai/mxbai-embed-large-v1"},
]

def recall_at_k(retrieved_ids, relevant_id, k):
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0

def mrr_score(retrieved_ids, relevant_id):
    for i, rid in enumerate(retrieved_ids):
        if rid == relevant_id:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_config(cfg):
    name       = cfg["name"]
    chunk_size = cfg["chunk_size"]
    overlap    = cfg["overlap"]
    model      = cfg["model"]

    chunk_path = f"data/chunks/ablation_{name}.jsonl"
    emb_path   = f"data/embeddings/ablation_{name}.npy"
    meta_path  = f"data/embeddings/ablation_{name}_meta.jsonl"
    text_path  = f"data/embeddings/ablation_{name}_texts.jsonl"
    collection = f"ablation_{name}"

    print(f"\n{'='*50}")
    print(f"Config: {name}")
    print(f"  chunk_size={chunk_size}, overlap={overlap}")
    print(f"  model={model}")
    print(f"{'='*50}")

    if not os.path.exists(chunk_path):
        print("Chunking...")
        chunk_corpus(
            in_path="data/raw/wiki.jsonl",
            out_path=chunk_path,
            chunk_size=chunk_size,
            overlap=overlap
        )
    else:
        print("Chunks already exist, skipping.")

    # Skip embedding if already done
    if not os.path.exists(emb_path):
        print("Embedding...")
        embed_corpus(
            chunk_path=chunk_path,
            emb_path=emb_path,
            meta_path=meta_path,
            text_path=text_path,
            model_name=model
        )
    else:
        print("Embeddings already exist, skipping.")

    print("Indexing...")
    build_index(
        emb_path=emb_path,
        meta_path=meta_path,
        text_path=text_path,
        collection_name=collection
    )

    # Evaluate
    print("Evaluating...")
    rag = RAGPipeline(
        embed_model=model,
        collection_name=collection
    )
    examples = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8")]

    scores = {f"recall@{k}": [] for k in KS}
    scores["mrr"] = []

    from tqdm import tqdm
    for ex in tqdm(examples, desc=f"Evaluating {name}"):
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
    results["config"] = cfg
    return results

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    all_results = []

    for cfg in CONFIGS:
        result = evaluate_config(cfg)
        all_results.append(result)
        print(f"\nResults for {cfg['name']}:")
        for k, v in result.items():
            if k != "config":
                print(f"  {k:12s}: {v:.4f}")

    with open(RESULTS_OUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {RESULTS_OUT}")