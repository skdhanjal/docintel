import json, pathlib
import psycopg
from pgvector.psycopg import register_vector
from docintel.ingest.parents import parent_key

CONN_STR = "postgresql://docintel:docintel_dev@localhost/docintel"

def load_chunks_file(conn, jsonl_path: pathlib.Path) -> int:
    """Insert one embeddings JSONL file's records into chunks.
    This is the real unit of work -- scoped to exactly one file, so
    an incremental single-document ingest (Day 24) never has to
    touch anything belonging to any other document."""
    records = [json.loads(l) for l in jsonl_path.read_text().strip().splitlines()]
    for r in records:
        headings = r.get("headings", [])
        pk = parent_key(r["doc_id"], headings) if headings else None
        conn.execute("""
            INSERT INTO chunks
                (chunk_id, doc_id, tenant_id, kind, page,
                 headings, parent_key,
                 text, token_count,
                 embed_model, embedding)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (chunk_id) DO NOTHING
        """, (
            r["chunk_id"], r["doc_id"], "default",
            r["kind"], r.get("page"),
            headings, pk,
            r["text"], r.get("token_count", 0),
            r["embed_model"], r["embedding"],
        ))
    conn.commit()
    return len(records)

def load_chunks(jsonl_dir: pathlib.Path, model_suffix: str = "model_a"):
    """Bulk loader for the initial corpus load (§2.3's first run) --
    a thin loop over load_chunks_file() for every matching file in
    the directory. Day 24's incremental sync does NOT call this; it
    calls load_chunks_file() directly on the one file that changed."""
    conn = psycopg.connect(CONN_STR)
    register_vector(conn)
    for f in sorted(jsonl_dir.glob(f"*.{model_suffix}.jsonl")):
        n = load_chunks_file(conn, f)
        print(f"loaded {n} chunks from {f.name}")
    conn.close()

def load_parents_for_doc(conn, doc_id: str):
    """Rebuild parent_docs rows for exactly one doc_id's chunks --
    the scoped counterpart to load_parents() below. Called after
    load_chunks_file() so a single-document ingest leaves parent_docs
    consistent without touching any other document's parents."""
    rows = conn.execute("""
        SELECT parent_key, doc_id, string_agg(text, E'\n\n' ORDER BY page, chunk_id)
        FROM chunks WHERE doc_id = %s AND parent_key IS NOT NULL
        GROUP BY parent_key, doc_id
    """, (doc_id,)).fetchall()
    for pk, d, txt in rows:
        conn.execute("""
            INSERT INTO parent_docs (parent_key, doc_id, text)
            VALUES (%s, %s, %s)
            ON CONFLICT (parent_key) DO UPDATE SET text = EXCLUDED.text
        """, (pk, d, txt))
    conn.commit()
    return len(rows)

def load_parents():
    """Bulk parent_docs build for the initial corpus load -- one
    GROUP BY across every doc's chunks. Day 24 uses the scoped
    load_parents_for_doc() instead, for the same reason as above."""
    conn = psycopg.connect(CONN_STR)
    rows = conn.execute("""
        SELECT parent_key, doc_id, string_agg(text, E'\n\n' ORDER BY page, chunk_id)
        FROM chunks WHERE parent_key IS NOT NULL
        GROUP BY parent_key, doc_id
    """).fetchall()
    for pk, doc_id, txt in rows:
        conn.execute("""
            INSERT INTO parent_docs (parent_key, doc_id, text)
            VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
        """, (pk, doc_id, txt))
    conn.commit()
    print(f"loaded {len(rows)} parent docs")
    conn.close()

if __name__ == "__main__":
    d = pathlib.Path("embeddings")
    load_chunks(d, "model_a")
    load_parents()
    # Quick count sanity check
    conn = psycopg.connect(CONN_STR)
    n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    p = conn.execute("SELECT count(*) FROM parent_docs").fetchone()[0]
    print(f"total: {n} chunks, {p} parents")
    conn.close()