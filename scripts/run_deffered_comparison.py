from docintel.gateway_client import GatewayClient
from docintel.retrieval.hybrid import hybrid_search, search_dense
from docintel.retrieval.pipeline import retrieve
from docintel.eval.retrieval_metrics import score_query
import json

client = GatewayClient()
with open("golden_set.json") as f:
    golden = json.load(f)

def avg(fn, k=10):
    scores = []
    for g in golden:
        retrieved_ids = fn(g["question"])
        scores.append(score_query(retrieved_ids, g["relevant_chunk_ids"], k=k))
    return {mk: round(sum(s[mk] for s in scores) / len(scores), 3) for mk in scores[0]}

# Comparison 1 — Day 22: embedding model A vs B
# (swap EMBED_MODEL in docintel/config.py between runs, re-embed corpus each time)
print("Embedding model (see DECISIONS.md for which model is active):",
      avg(lambda q: [r["chunk_id"] for r in search_dense(q, client, k=10, tenant_id="tenant_alpha")]))

# Comparison 2 — Day 25: hybrid vs dense-only
print("Dense-only:", avg(lambda q: [r["chunk_id"] for r in search_dense(q, client, k=10, tenant_id="tenant_alpha")]))
print("Hybrid (RRF):", avg(lambda q: [r["chunk_id"] for r in hybrid_search(q, client, k=10, tenant_id="tenant_alpha")]))

# Comparison 3 — Day 26: rerank on vs off
print("Rerank OFF:", avg(lambda q: [r["chunk_id"] for r in retrieve(q, client, top_n=10, use_rerank=False, tenant_id="tenant_alpha")[0]]))
print("Rerank ON: ", avg(lambda q: [r["chunk_id"] for r in retrieve(q, client, top_n=10, use_rerank=True,  tenant_id="tenant_alpha")[0]]))