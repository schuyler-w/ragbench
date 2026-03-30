import chromadb, json, numpy as np
from tqdm import tqdm

EMB_PATH   = "data/embeddings/embeddings.npy"
META_PATH  = "data/embeddings/metadata.jsonl"
TEXT_PATH  = "data/embeddings/texts.jsonl"
CHROMA_DIR = "./chroma_db"
COLLECTION = "wikipedia"
BATCH_SIZE = 5000

def build_index(emb_path=EMB_PATH, meta_path=META_PATH, text_path=TEXT_PATH, collection_name=COLLECTION):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print("Loading embeddings...")
    embeddings = np.load(emb_path)

    metadata, texts = [], []
    with open(meta_path, encoding="utf-8") as fm, open(text_path, encoding="utf-8") as ft: 
        for ml, tl in zip(fm, ft):
            metadata.append(json.loads(ml))
            texts.append(json.loads(tl))

    ids = [m["chunk_id"] for m in metadata]
    total = len(ids)
    print(f"Indexing {total:,} chunks...")

    for i in tqdm(range(0, total, BATCH_SIZE)):
        collection.add(
            ids=        ids[i:i+BATCH_SIZE],
            embeddings= embeddings[i:i+BATCH_SIZE].tolist(),
            documents=  texts[i:i+BATCH_SIZE],
            metadatas=  metadata[i:i+BATCH_SIZE]
        )

    print(f"Index complete. Collection size: {collection.count():,} chunks")

if __name__ == "__main__":
    build_index()