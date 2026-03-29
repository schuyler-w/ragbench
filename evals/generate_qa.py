import json, os, random, requests
from tqdm import tqdm

CHUNK_PATH   = "data/chunks/wiki_chunks.jsonl"
OUT_PATH     = "evals/eval_set.jsonl"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:70b"
N_EXAMPLES   = 500
SEED         = 101

PROMPT = """Read the passage below and write ONE specific factual question that:
- Can be answered using ONLY the passage
- Has a clear, unambiguous answer
- Is not a yes/no question

Output only the question. No preamble.

Passage:
{text}

Question:"""

def generate_question(text: str) -> str:
    r = requests.post(OLLAMA_URL, json={
        "model":   OLLAMA_MODEL,
        "prompt":  PROMPT.format(text=text[:1500]),
        "stream":  False,
        "options": {"temperature": 0.7}
    }, timeout=60)
    return r.json()["response"].strip()

def build_eval_set():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    all_chunks = []
    with open(CHUNK_PATH, encoding="utf-8") as f:
        for line in f:
            all_chunks.append(json.loads(line))

    random.seed(SEED)
    sample = random.sample(all_chunks, N_EXAMPLES)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for chunk in tqdm(sample, desc="Generating questions"):
            try:
                question = generate_question(chunk["text"])
                record = {
                    "question":          question,
                    "relevant_chunk_id": chunk["chunk_id"],
                    "relevant_doc_id":   chunk["doc_id"],
                    "source_title":      chunk["title"],
                    "gold_text":         chunk["text"]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Skipping chunk {chunk['chunk_id']}: {e}")

    print(f"Eval set saved to {OUT_PATH}")

if __name__ == "__main__":
    build_eval_set()