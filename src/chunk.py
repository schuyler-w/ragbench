import json, os, tiktoken
from tqdm import tqdm

CHUNK_SIZE    = 256
CHUNK_OVERLAP = 32

enc = tiktoken.get_encoding("cl100k_base")

def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks

def chunk_corpus(
    in_path="data/raw/wiki.jsonl",
    out_path="data/chunks/wiki_chunks.jsonl",
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total_chunks = 0

    with open(in_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Chunking"):
            doc = json.loads(line)
            chunks = split_into_chunks(doc["text"], chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                record = {
                    "chunk_id":    f"{doc['id']}_{i}",
                    "doc_id":      doc["id"],
                    "title":       doc["title"],
                    "source":      doc["source"],
                    "chunk_index": i,
                    "text":        chunk
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Total chunks written: {total_chunks:,}")
    return total_chunks

if __name__ == "__main__":
    chunk_corpus()