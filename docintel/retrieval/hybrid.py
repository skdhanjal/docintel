import psycopg
from pgvector.psycopg import register_vector
from docintel.gateway_client import GatewayClient
from docintel.retrieval.rrf import reciprocal_rank_fusion
from docintel.ingest.load_db import CONN_STR

# docintel/retrieval/hybrid.py

def set_tenant_context(conn, tenant_id: str):
    # is_local = True corresponds to "SET LOCAL" (scoped to the current transaction)
    conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))

def search_dense(conn, qvec, tenant_id: str, k: int = 20, acl_groups: list[str] | None = None):
    """ANN search via pgvector HNSW. tenant_id has NO default — every
    call site must pass one explicitly. acl_groups, if given, further
    restricts to chunks visible to the caller's role(s)."""
    set_tenant_context(conn, tenant_id)
    acl_clause, params = "", (qvec, tenant_id, qvec, k)
    if acl_groups:
        acl_clause = "AND acl_groups && %s"
        params = (qvec, tenant_id, acl_groups, qvec, k)
    rows = conn.execute(f"""
        SELECT chunk_id, doc_id, kind, page, headings,
               text, token_count, parent_key,
               embedding <=> %s::vector AS dist
        FROM chunks
        WHERE tenant_id = %s {acl_clause}
        ORDER BY embedding <=> %s::vector LIMIT %s
    """, params).fetchall()
    return rows

def search_sparse(conn, query_text: str, tenant_id: str, k: int = 20, acl_groups: list[str] | None = None):
    """Full-text search via tsvector + ts_rank_cd. Same treatment as
    search_dense: tenant_id has no default, RLS context is set first,
    and the optional acl_groups filter is identical in shape."""
    set_tenant_context(conn, tenant_id)
    acl_clause, params = "", (query_text, query_text, tenant_id, k)
    
    if acl_groups:
        acl_clause = "AND acl_groups && %s"
        params = (query_text, query_text, tenant_id, acl_groups, k)
        
    rows = conn.execute(f"""
        SELECT chunk_id, doc_id, kind, page, headings,
               text, token_count, parent_key,
               ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS rank
        FROM chunks
        WHERE tsv @@ plainto_tsquery('english', %s) AND tenant_id = %s {acl_clause}
        ORDER BY rank DESC LIMIT %s
    """, params).fetchall()
    
    return rows

def hybrid_search(
    query: str,
    client,
    tenant_id: str,
    k: int = 10,
    dense_k: int = 20,
    sparse_k: int = 20,
    acl_groups: list[str] | None = None,
) -> list[dict]:
    """Same fusion logic as Day 25 — the only change is that tenant_id
    is now a required positional argument (no default) threaded through
    to both search_dense and search_sparse, and acl_groups passes
    through identically to both. There is no code path in this
    function that can run an unfiltered query."""
    conn = psycopg.connect(CONN_STR)
    register_vector(conn)
    conn.execute("SET hnsw.ef_search = 64")

    qvec = client.embed([query])[0]
    dense_rows = search_dense(conn, qvec, tenant_id, dense_k, acl_groups)
    sparse_rows = search_sparse(conn, query, tenant_id, sparse_k, acl_groups)

    dense_ids  = [r[0] for r in dense_rows]
    sparse_ids = [r[0] for r in sparse_rows]
    fused = reciprocal_rank_fusion(dense_ids, sparse_ids)

    all_rows = {r[0]: r for r in dense_rows + sparse_rows}
    results = []
    for cid, rrf_score in fused[:k]:
        r = all_rows.get(cid)
        if not r:
            continue
        results.append({
            "chunk_id": r[0], "doc_id": r[1], "kind": r[2],
            "page": r[3], "headings": r[4],
            "text": r[5], "token_count": r[6], "parent_key": r[7],
            "rrf_score": rrf_score,
            "in_dense": cid in dense_ids,
            "in_sparse": cid in sparse_ids,
        })
    conn.close()
    return results

if __name__ == "__main__":
    client = GatewayClient()
    results = hybrid_search("Item 7A market risk disclosures", client)
    print(f"hybrid results: {len(results)}")
    for r in results[:5]:
        src = "D+S" if r["in_dense"] and r["in_sparse"] else ("D" if r["in_dense"] else "S")
        print(f"  rrf={r['rrf_score']:.4f}  [{src}]  {r['kind']:6s}  "
              f"{r['doc_id'][:14]:14s}  p.{r['page']}  {r['text'][:20]}…")