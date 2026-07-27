import json, pathlib, numpy as np
from docintel.gateway_client import GatewayClient

# Load all Model A embeddings into memory (small corpus).
recs = []
for f in pathlib.Path("embeddings").glob("*.model_a.jsonl"):
    recs += [json.loads(l) for l in f.read_text().strip().splitlines()]

vecs = np.array([r["embedding"] for r in recs], dtype=np.float32)
norms = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

client = GatewayClient()

QUERIES = [
    "What was total revenue in fiscal 2023?",
    "What are the main risk factors?",
    "What was R&D spend?",
    "Describe the debt maturity schedule.",
    "Item 7A market risk disclosures",
]

for q in QUERIES:
    qvec = np.array(client.embed([q])[0], dtype=np.float32)
    qvec /= np.linalg.norm(qvec)
    sims = norms @ qvec
    top5 = np.argsort(-sims)[:5]
    print(f"\n--- {q} ---")
    for idx in top5:
        r = recs[idx]
        print(f"  sim={sims[idx]:.3f}  {r['kind']:6s}  {r['doc_id'][:12]:12s}  "
              f"p.{r.get('page','?')}  {r['text'][:80]}…")