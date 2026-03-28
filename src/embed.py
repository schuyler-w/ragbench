import json, os, numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME  = "BAAI/bge-large-en-v1.5"
BATCH_SIZE  = 512
DEVICE      = "cuda"
CHUNK_PATH  = "data/chunks/wiki_chunks.jsonl"
EMB_PATH    = "data/embeddings/embeddings.npy"
META_PATH   = "data/embeddings/metadata.jsonl"
TEXT_PATH   = "data/embeddings/texts.jsonl"

def embed_corpus(
    chunk_path=CHUNK_PATH,
    emb_path=EMB_PATH,
    meta_path=META_PATH,
    text_path=TEXT_PATH,
    model_name=MODEL_NAME,
    batch_size=BATCH_SIZE
):
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=DEVICE)

    texts, metadata = [], []
    print("Reading chunks...")
    with open(chunk_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            record = json.loads(line)
            texts.append(record["text"])
            metadata.append({k: v for k, v in record.items() if k != "text"})

    print(f"Embedding {len(texts):,} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=DEVICE
    )

    np.save(emb_path, embeddings)
    print(f"Saved embeddings: {embeddings.shape}")

    with open(meta_path, "w", encoding="utf-8") as fm:
        for m in metadata:
            fm.write(json.dumps(m) + "\n")

    with open(text_path, "w", encoding="utf-8") as ft:
        for t in texts:
            ft.write(json.dumps(t) + "\n")

    print("Done.")

if __name__ == "__main__":
    embed_corpus()