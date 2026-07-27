import json, pathlib

for f in sorted(pathlib.Path("embeddings").glob("*.jsonl")):
    records = [json.loads(l) for l in f.read_text().strip().splitlines()]
    models = set(r["embed_model"] for r in records)
    dims   = set(r["embed_dims"]  for r in records)
    kinds  = {r["kind"] for r in records}
    assert len(models) == 1, f"mixed models in {f.name}!"
    assert len(dims) == 1,   f"mixed dims in {f.name}!"
    print(f"{f.name:40s}  records={len(records):5d}  "
          f"model={models.pop():30s}  dims={dims.pop()}  "
          f"kinds={kinds}")

print("\n✓ No mixed-model contamination.")