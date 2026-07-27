import time
from docintel.retrieval.hybrid import hybrid_search
from docintel.retrieval.rerank import rerank

def retrieve(
    query: str,
    client,
    top_n: int = 5,
    candidate_k: int = 50,
    use_rerank: bool = True,
    tenant_id: str = "default",
) -> tuple[list[dict], dict]:
    """Two-stage retrieval: hybrid fetch of candidate_k, optional rerank to top_n.

    Returns (results, timing). timing has retrieve_ms / rerank_ms / total_ms
    so both configurations can be measured and compared, not asserted.
    """
    t0 = time.perf_counter()
    candidates = hybrid_search(
        query, client, k=candidate_k,
        dense_k=candidate_k, sparse_k=candidate_k, tenant_id=tenant_id,
    )
    t1 = time.perf_counter()

    if use_rerank:
        results = rerank(query, candidates, top_n=top_n)
    else:
        # Without reranking, RRF order is the final order — just truncate.
        results = candidates[:top_n]
    t2 = time.perf_counter()

    timing = {
        "retrieve_ms": (t1 - t0) * 1000,
        "rerank_ms": (t2 - t1) * 1000 if use_rerank else 0.0,
        "total_ms": (t2 - t0) * 1000,
        "use_rerank": use_rerank,
    }
    return results, timing

if __name__ == "__main__":
    from docintel.gateway_client import GatewayClient
    client = GatewayClient()
    query = "How did gross margin trend?"

    for flag in (True, False):
        results, timing = retrieve(query, client, top_n=5, use_rerank=flag)
        label = "rerank=ON " if flag else "rerank=OFF"
        print(f"\n{label}  total={timing['total_ms']:.0f}ms  "
              f"(retrieve={timing['retrieve_ms']:.0f}ms, rerank={timing['rerank_ms']:.0f}ms)")
        for i, r in enumerate(results, 1):
            score = r.get("rerank_score", r.get("rrf_score"))
            print(f"  #{i}  {score:.3f}  {r['text'][:70]}…")