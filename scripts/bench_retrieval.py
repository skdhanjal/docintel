import time, statistics
import psycopg
from pgvector.psycopg import register_vector
from docintel.gateway_client import GatewayClient

CONN_STR = "postgresql://docintel:docintel_dev@localhost/docintel"
K = 10

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

# Set ef_search for this session.
conn.execute("SET hnsw.ef_search = 64")

def search(qvec, where: str = "", params: tuple = ()):
    sql = f"""
        SELECT chunk_id, doc_id, kind, page, text,
               embedding <=> %s::vector AS dist
        FROM chunks {where}
        ORDER BY embedding <=> %s::vector
        LIMIT {K}
    """
    t0 = time.perf_counter()
    # Correct order: 1st vector, then WHERE params, then 2nd vector
    rows = conn.execute(sql, (qvec,) + params + (qvec,)).fetchall()
    dt = (time.perf_counter() - t0) * 1000
    return rows, dt

print(f"{'Query':40s}  {'Unfilt ms':>10s}  {'Tenant ms':>10s}  {'Tables ms':>10s}")
print("-"*78)

for q in QUERIES:
    qvec = client.embed([q])[0]
    _, t_none  = search(qvec)
    _, t_tenant = search(qvec, "WHERE tenant_id = %s", ("default",))
    _, t_table  = search(qvec, "WHERE kind = %s", ("table",))
    print(f"{q[:38]:38s}  {t_none:10.2f}  {t_tenant:10.2f}  {t_table:10.2f}")

# Also print top results for one query so you can eyeball quality.
qvec = client.embed(["What was R&D spend?"])[0]
rows, _ = search(qvec)
print("\n--- top-5 for 'What was R&D spend?' ---")
for r in rows[:5]:
    print(f"  dist={r[5]:.3f}  {r[2]:6s}  {r[1][:14]:14s}  p.{r[3]}  {r[4][:60]}…")

conn.close()