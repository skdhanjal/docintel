import pathlib
import psycopg
from docintel.config import settings
from docintel.ingest.manifest import load_manifest, save_manifest, scan_corpus
from docintel.ingest.diff import compute_diff
from docintel.ingest.parse_pipeline import parse_and_cache
from docintel.ingest.chunker import chunk_document
from docintel.ingest.embed import embed_chunks, save_embeddings
from docintel.ingest.load_db import load_chunks_file, load_parents_for_doc, CONN_STR
from pgvector.psycopg import register_vector
from docintel.gateway_client import GatewayClient

CORPUS = pathlib.Path(settings.corpus_dir)
PARSED = pathlib.Path("parsed")
EMBEDS = pathlib.Path("embeddings")

def clean_artifacts(filename: str):
    """Remove EVERY on-disk artifact for a file: the parsed cache
    AND the embeddings JSONL, not just one of the two. Called before
    delete_doc() removes DB rows, and again before ingest_file()
    re-creates them on update -- so no stage of the pipeline can
    ever silently serve bytes left over from a previous version of
    the document. This is the fix for two separate gaps: the delete
    phase used to clean only the parsed cache, and the update phase
    didn't clean anything at all before re-ingesting."""
    stem = pathlib.Path(filename).stem
    for p in PARSED.glob(f"{stem}.*"):
        p.unlink(); print(f"  removed parsed cache {p.name}")
    for p in EMBEDS.glob(f"{filename}.*.jsonl"):
        p.unlink(); print(f"  removed embeddings {p.name}")

def delete_doc(doc_id: str):
    """Remove all chunks and parents for a doc from pgvector.
    parent_docs cleanup joins on parent_key, matching the scheme
    Day 23's load_db.py uses to populate it -- the same key that
    was never persisted on the Chunk object itself (Day 21 §2.2),
    computed identically wherever it's needed."""
    conn = psycopg.connect(CONN_STR)
    n = conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,)).rowcount
    conn.execute("""DELETE FROM parent_docs WHERE parent_key IN
        (SELECT DISTINCT parent_key FROM chunks WHERE doc_id = %s)""", (doc_id,))
    conn.commit(); conn.close()
    print(f"  deleted {n} chunks for {doc_id}")

def ingest_file(filename: str, client: GatewayClient):
    """Full pipeline for one file: parse+cache -> chunk -> embed -> load.
    chunk_document() already produces contextualized text (Day 21) --
    no separate enrichment pass needed before embedding.

    Loads and parent-rebuilds are scoped to THIS file only, via
    load_chunks_file() / load_parents_for_doc() -- not the bulk
    load_chunks()/load_parents() from Day 23, which would re-scan
    and re-insert every other document's embeddings on every single
    incremental ingest. A 3-file update should cost O(3), not O(corpus)."""
    src = CORPUS / filename
    parsed_path = parse_and_cache(src, PARSED)
    cks = chunk_document(filename, PARSED / f"{pathlib.Path(filename).stem}.json")
    recs = embed_chunks(cks, client)
    out = EMBEDS / f"{filename}.model_a.jsonl"
    save_embeddings(recs, out)

    conn = psycopg.connect(CONN_STR)
    register_vector(conn)
    n_loaded = load_chunks_file(conn, out)
    n_parents = load_parents_for_doc(conn, filename)
    conn.close()

    print(f"  ingested {filename}: {n_loaded} chunks, {n_parents} parent sections")

def sync():
    manifest = load_manifest()
    current = scan_corpus(CORPUS)
    diff = compute_diff(manifest, current)

    print(f"diff: add={len(diff.to_add)} update={len(diff.to_update)} "
          f"delete={len(diff.to_delete)} unchanged={len(diff.unchanged)}")

    if not (diff.to_add or diff.to_update or diff.to_delete):
        print("nothing to do — index is current.")
        return

    client = GatewayClient()

    # 1. Deletes first — remove DB rows AND every on-disk artifact.
    for name in diff.to_delete:
        print(f"DELETE {name}:")
        delete_doc(name)
        clean_artifacts(name)

    # 2. Updates — delete DB rows + old artifacts, THEN re-ingest clean.
    #    Cleaning before re-ingesting (not just relying on ingest_file's
    #    writes to overwrite same-named files) means a failure partway
    #    through re-ingestion can never leave stale bytes mixed with
    #    new ones -- delete-then-create, not overwrite-and-hope.
    for name in diff.to_update:
        print(f"UPDATE {name}:")
        delete_doc(name)
        clean_artifacts(name)
        ingest_file(name, client)

    # 3. Adds — full pipeline for each new file.
    for name in diff.to_add:
        print(f"ADD {name}:")
        ingest_file(name, client)

    # 4. Save updated manifest.
    save_manifest(current)
    print("✓ manifest updated.")

if __name__ == "__main__":
    sync()