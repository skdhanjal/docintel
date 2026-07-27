import pathlib, random
from docintel.ingest.chunker import chunk_document

p = next(pathlib.Path("parsed").glob("*.json"))
cks = chunk_document(p.stem, p)

random.seed(21)
sample = random.sample(cks, min(20, len(cks)))

for i, c in enumerate(sample, 1):
    print(f"\n{'='*60}")
    print(f"CHUNK {i}/{len(sample)} · {c.kind} · page {c.page} · {c.token_count} tokens")
    print(f"headings: {' → '.join(c.headings) or '(root)'}")
    print(f"---")
    print(c.text[:400])
    print("OK [ ] ISSUE: ____")