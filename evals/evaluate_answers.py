import json, os, requests, numpy as np
from tqdm import tqdm

EVAL_PATH    = "evals/eval_set.jsonl"
RESULTS_OUT  = "results/answer_metrics.json"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
N            = 100

def ollama(prompt: str, temperature: float = 0.0) -> str:
    try:
        r = requests.post(OLLAMA_URL, json={
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": temperature}
        }, timeout=300)
        return r.json()["response"].strip()
    except Exception as e:
            print(f"Ollama error: {e}")
            return ""

# --- Metric 1: Faithfulness ---
FAITHFULNESS_EXTRACT = """Break the following answer into a list of atomic factual claims.
Output one claim per line. No bullet points, no numbers, no preamble.
If there are no claims, output "NONE".

Answer: {answer}

Claims:"""

FAITHFULNESS_CHECK = """Given the context below, is the following claim directly supported by the context?
Answer with only YES or NO.

Context:
{context}

Claim: {claim}

Answer:"""

def faithfulness_score(answer: str, contexts: list) -> float:
    context = "\n\n".join(contexts)

    claims_raw = ollama(FAITHFULNESS_EXTRACT.format(answer=answer))
    if "NONE" in claims_raw or not claims_raw.strip():
        return 1.0
    claims = [c.strip() for c in claims_raw.strip().split("\n") if c.strip()]

    if not claims:
        return 1.0

    supported = 0
    for claim in claims:
        verdict = ollama(FAITHFULNESS_CHECK.format(context=context, claim=claim))
        if verdict.upper().startswith("YES"):
            supported += 1

    return supported / len(claims)

# --- Metric 2: Answer Relevancy ---
RELEVANCY_PROMPT = """Given the question and answer below, rate how well the answer
addresses the question on a scale of 0 to 1, where:
1.0 = fully answers the question
0.5 = partially answers the question
0.0 = does not answer the question at all

Output only a number between 0 and 1. No explanation.

Question: {question}
Answer: {answer}

Score:"""

def answer_relevancy_score(question: str, answer: str) -> float:
    raw = ollama(RELEVANCY_PROMPT.format(question=question, answer=answer))
    try:
        score = float(raw.strip().split()[0])
        return max(0.0, min(1.0, score))
    except:
        return 0.0

# --- Metric 3: Context Recall ---
CONTEXT_RECALL_PROMPT = """Given the question and context below, does the context contain
sufficient information to answer the question?
Answer with only YES or NO.

Question: {question}

Context:
{context}

Answer:"""

def context_recall_score(question: str, contexts: list) -> float:
    context = "\n\n".join(contexts)
    verdict = ollama(CONTEXT_RECALL_PROMPT.format(question=question, context=context))
    return 1.0 if verdict.upper().startswith("YES") else 0.0

def evaluate():
    os.makedirs("results", exist_ok=True)

    from src.rag import RAGPipeline
    rag = RAGPipeline()
    examples = [json.loads(l) for l in open(EVAL_PATH, encoding="utf-8")][:N]

    scores = {
        "faithfulness":      [],
        "answer_relevancy":  [],
        "context_recall":    []
    }

    for ex in tqdm(examples, desc="Evaluating answers"):
        retrieved  = rag.retrieve(ex["question"])
        answer     = rag.generate(ex["question"], retrieved)
        contexts   = [c for c, _, _ in retrieved]

        scores["faithfulness"].append(
            faithfulness_score(answer, contexts)
        )
        scores["answer_relevancy"].append(
            answer_relevancy_score(ex["question"], answer)
        )
        scores["context_recall"].append(
            context_recall_score(ex["question"], contexts)
        )

    results = {metric: round(float(np.mean(vals)), 4)
               for metric, vals in scores.items()}

    print("\n=== Answer Evaluation Results ===")
    for metric, val in results.items():
        print(f"  {metric:20s}: {val:.4f}")

    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_OUT}")
    return results

if __name__ == "__main__":
    evaluate()