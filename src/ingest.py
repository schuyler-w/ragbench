from datasets import load_dataset
from tqdm import tqdm
import json, os

def ingest_wikipedia(n_docs=50_000, out_path="data/raw/wiki.jsonl"):
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}. Delete it to re-ingest.")
        return

    print(f"Streaming Wikipedia... saving {n_docs} docs")
    dataset_name = "wikimedia/wikipedia"
    ds = load_dataset(
        dataset_name,
        "20231101.en",
        split="train",
        streaming=True,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    saved = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for doc in tqdm(ds, total=n_docs, desc="Ingesting"):
            if saved >= n_docs:
                break
            if len(doc["text"]) < 500:
                continue
            record = {
                "id":     doc["id"],
                "title":  doc["title"],
                "text":   doc["text"],
                "source": "wikipedia"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1

    print(f"Done. Saved {saved} docs to {out_path}")

if __name__ == "__main__":
    ingest_wikipedia()