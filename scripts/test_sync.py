import pathlib, shutil, json, time
import psycopg
from docintel.ingest.sync import sync, CONN_STR
from docintel.config import settings

CORPUS = pathlib.Path(settings.corpus_dir)

def count_chunks(doc_id: str | None = None) -> int:
    conn = psycopg.connect(CONN_STR)
    if doc_id:
        n = conn.execute("SELECT count(*) FROM chunks WHERE doc_id=%s",
                         (doc_id,)).fetchone()[0]
    else:
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    conn.close()
    return n

# --- Test 1: Idempotency (no change) ---
print("TEST 1: No change — should be a no-op")
before = count_chunks()
sync()
after = count_chunks()
assert before == after, f"chunk count changed! {before}→{after}"
print(f"  ✓ {before} chunks, unchanged\n")

# --- Test 2: Add a new file ---
print("TEST 2: Add — create a test file")
test_file = CORPUS / "TEST_DOC.html"
test_file.write_text("<html><body><h1>Test Filing</h1><p>Revenue was $999.</p></body></html>")
sync()
n = count_chunks("TEST_DOC.html")
assert n > 0, "no chunks created for TEST_DOC!"
print(f"  ✓ {n} chunks added for TEST_DOC.html\n")

# --- Test 3: Update the file ---
print("TEST 3: Update — modify the test file")
test_file.write_text("<html><body><h1>Updated Filing</h1><p>Revenue was $1,234.</p></body></html>")
sync()
n2 = count_chunks("TEST_DOC.html")
assert n2 > 0, "no chunks after update!"
print(f"  ✓ {n2} chunks after update (was {n})\n")

# --- Test 4: Delete the file ---
print("TEST 4: Delete — remove the test file")
test_file.unlink()
sync()
n3 = count_chunks("TEST_DOC.html")
assert n3 == 0, f"ghost chunks remain! count={n3}"
print(f"  ✓ 0 chunks — ghost chunks eliminated\n")

# --- Final ---
total = count_chunks()
print(f"ALL TESTS PASSED. Total chunks in store: {total}")