import chromadb, requests
from sentence_transformers import SentenceTransformer

EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:70b"
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "wikipedia"
TOP_K        = 5
DEVICE       = "cuda"

PROMPT_TEMPLATE = """You are a factual assistant. Answer the question using ONLY the context below.
If the answer is not present in the context, respond with "I don't know."
Do not add information from outside the context.

Context:
{context}

Question: {question}
Answer:"""

class RAGPipeline:
    def __init__(self,
             embed_model=EMBED_MODEL,
             ollama_model=OLLAMA_MODEL,
             collection_name=COLLECTION):
        print(f"Loading embedding model: {embed_model}")
        self.embedder = SentenceTransformer(embed_model, device=DEVICE)
        self.ollama_model = ollama_model
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(collection_name)
        print(f"Connected to collection: {collection_name} ({self.collection.count():,} chunks)")

    def retrieve(self, query: str, k: int = TOP_K):
        query_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            device=DEVICE
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        chunks = results["documents"][0]
        metas  = results["metadatas"][0]
        scores = results["distances"][0]
        return list(zip(chunks, metas, scores))

    def generate(self, question: str, retrieved: list) -> str:
        context = "\n\n---\n\n".join([chunk for chunk, _, _ in retrieved])
        prompt  = PROMPT_TEMPLATE.format(context=context, question=question)

        response = requests.post(OLLAMA_URL, json={
            "model":   self.ollama_model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.1}
        }, timeout=120)
        response.raise_for_status()
        return response.json()["response"].strip()

    def query(self, question: str, k: int = TOP_K) -> dict:
        retrieved = self.retrieve(question, k)
        answer    = self.generate(question, retrieved)
        return {
            "question": question,
            "answer":   answer,
            "sources": [
                {
                    "title":   m["title"],
                    "score":   round(s, 4),
                    "snippet": c[:200]
                }
                for c, m, s in retrieved
            ]
        }

if __name__ == "__main__":
    rag = RAGPipeline()
    result = rag.query("What is the speed of light?")
    print(f"\nAnswer: {result['answer']}\n")
    print("Sources:")
    for src in result["sources"]:
        print(f"  [{src['score']:.4f}] {src['title']}")