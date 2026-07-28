import json, random
import psycopg
from docintel.gateway_client import GatewayClient

CONN_STR = "postgresql://docintel:docintel_dev@localhost/docintel"
N_SYNTHETIC = 50

QUESTION_PROMPT = """Write ONE specific question that this passage directly and \
completely answers. The question should sound like something a real \
analyst would type — not "What does this passage say about X." \
Return only the question, nothing else.

PASSAGE:
{text}"""

def sample_chunks(n: int) -> list[dict]:
    conn = psycopg.connect(CONN_STR)
    # Bias toward narrative/table chunks with enough content to
    # support a well-formed question; skip tiny fragments.
    rows = conn.execute("""
        SELECT chunk_id, doc_id, kind, page, text FROM chunks
        WHERE length(text) > 200 ORDER BY random() LIMIT %s
    """, (n,)).fetchall()
    conn.close()
    return [{"chunk_id": r[0], "doc_id": r[1], "kind": r[2], "page": r[3], "text": r[4]} for r in rows]

def generate_synthetic_set() -> list[dict]:
    client = GatewayClient()
    chunks = sample_chunks(N_SYNTHETIC)
    golden = []
    for c in chunks:
        question = client.complete(QUESTION_PROMPT.format(text=c["text"][:1200]), max_tokens=80)
        golden.append({
            "question": question.strip(),
            "relevant_chunk_ids": [c["chunk_id"]],
            "archetype": "factual" if c["kind"] == "text" else "table-lookup",
            "source": "synthetic",
            "reviewed": False,
        })
    return golden

if __name__ == "__main__":
    golden = generate_synthetic_set()
    with open("golden_set_synthetic.json", "w") as f:
        json.dump(golden, f, indent=2)
    print(f"Generated {len(golden)} synthetic questions → golden_set_synthetic.json")
    print("Sample for spot-check:")
    for g in random.sample(golden, 5):
        print(f"  - {g['question']}")