from docintel.gateway_client import GatewayClient
from docintel.retrieval.pipeline import retrieve

MULTI_QUERY_PROMPT = """Generate {n} different search queries that would help \
answer the question below. Vary vocabulary and phrasing — do not \
just add words. Return one query per line, no numbering, no extra text.

Question: {question}"""

DECOMPOSE_PROMPT = """Break the question below into 2-4 independent \
sub-questions, each answerable by searching a single document. \
Return one sub-question per line, no numbering, no extra text. \
If the question is already simple and single-document, return it \
unchanged as the only line.

Question: {question}"""

def multi_query_expand(question: str, client: GatewayClient, n: int = 3) -> list[str]:
    """Return [original, variant_1, ..., variant_n]. Original is always
    included — a bad LLM paraphrase should never fully replace the
    user's actual words."""
    prompt = MULTI_QUERY_PROMPT.format(n=n, question=question)
    raw = client.complete(prompt, model="cheap", max_tokens=200)
    variants = [q.strip("-• ").strip() for q in raw.split("\n") if q.strip()]
    return [question] + variants[:n]

def multi_query_retrieve(
    question: str, client: GatewayClient, top_n: int = 5, n_variants: int = 3,
) -> tuple[list[dict], list[str]]:
    """Retrieve for the original + each variant; dedup by chunk_id,
    keeping the highest score seen for each chunk across all queries."""
    queries = multi_query_expand(question, client, n=n_variants)
    seen: dict[str, dict] = {}
    for q in queries:
        results, _ = retrieve(q, client, top_n=top_n, use_rerank=True)
        for r in results:
            cid = r["chunk_id"]
            score = r.get("rerank_score", 0)
            if cid not in seen or score > seen[cid].get("rerank_score", 0):
                seen[cid] = r
    fused = sorted(seen.values(), key=lambda r: -r.get("rerank_score", 0))
    return fused[:top_n], queries

def decompose(question: str, client: GatewayClient) -> list[str]:
    raw = client.complete(DECOMPOSE_PROMPT.format(question=question), model="cheap", max_tokens=250)
    sub_qs = [q.strip("-• ").strip() for q in raw.split("\n") if q.strip()]
    return sub_qs or [question]

def decompose_retrieve(
    question: str, client: GatewayClient, top_n_per_sub: int = 3,
) -> tuple[list[dict], list[str]]:
    """Retrieve independently for each sub-question and concatenate —
    deliberately NOT deduped/fused like multi-query, because each
    sub-question targets a *different* document and losing any one
    means losing coverage of that entity entirely."""
    sub_qs = decompose(question, client)
    all_results = []
    for sq in sub_qs:
        results, _ = retrieve(sq, client, top_n=top_n_per_sub, use_rerank=True)
        all_results.extend(results)
    return all_results, sub_qs

if __name__ == "__main__":
    client = GatewayClient()
    results, queries = multi_query_retrieve("What was R&D spend?", client)
    print("Variants generated:")
    for q in queries:
        print(f"  - {q}")
    print(f"\nFused top-{len(results)}:")
    for r in results:
        print(f"  {r['text'][:70]}…")
        
    results, sub_qs = decompose_retrieve(
        "Compare R&D spending trends across the three tech filings.", client,
    )
    print("Sub-questions:")
    for sq in sub_qs:
        print(f"  - {sq}")
    print(f"\nDocs covered: {sorted({r['doc_id'] for r in results})}")    