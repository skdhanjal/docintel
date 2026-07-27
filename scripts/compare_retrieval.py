import psycopg
from pgvector.psycopg import register_vector
from docintel.gateway_client import GatewayClient
from docintel.retrieval.hybrid import search_dense, search_sparse, hybrid_search
from docintel.ingest.load_db import CONN_STR

QUERIES = [
    "What was total revenue in fiscal 2023?",
    "What are the main risk factors?",
    "What was R&D spend?",
    "Describe the debt maturity schedule.",
    "Item 7A market risk disclosures",
]

client = GatewayClient()
conn = psycopg.connect(CONN_STR)
register_vector(conn)
conn.execute("SET hnsw.ef_search = 64")

print(f"{'Query':42s}  {'Dense top-5':20s}  {'Sparse top-5':20s}  {'Hybrid top-5':20s}")
print("="*106)

for q in QUERIES:
    qvec = client.embed([q])[0]
    d = [r[0] for r in search_dense(conn, qvec, k=10)]
    s = [r[0] for r in search_sparse(conn, q, k=10)]
    h = hybrid_search(q, client, k=10)

    # Preview the text snippets for manual inspection.
    d_snip = "; ".join(r[5][:30] for r in search_dense(conn, qvec, k=3))
    s_snip = "; ".join(r[5][:30] for r in search_sparse(conn, q, k=3))
    h_snip = "; ".join(r["text"][:30] for r in h[:3])
    print(f"{q[:40]:40s}  {d_snip[:20]:20s}  {s_snip[:20]:20s}  {h_snip[:20]:20s}")

conn.close()
print("\nManually check: does the answer-bearing chunk appear in each column?")
print("Record Dense/Sparse/Hybrid recall for each query in DECISIONS.md.")