import pathlib, json
from dataclasses import asdict
from docintel.ingest.chunker import chunk_document

PARSED = pathlib.Path("parsed")
OUT = pathlib.Path("chunked"); 
OUT.mkdir(parents=True, exist_ok=True)

for jp in sorted(PARSED.glob("*.json")):
    cks = chunk_document(jp.stem, jp)
    (OUT / f"{jp.stem}.json").write_text(json.dumps([asdict(c) for c in cks], indent=2))
    n_tables = sum(1 for c in cks if c.kind == "table")
    print(f"{jp.stem:20s} {len(cks):4d} chunks   {n_tables:3d} tables")