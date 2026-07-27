import json, pathlib, random

PARSED = pathlib.Path("parsed")

def sample(n_per_file: int = 4, files: int = 3):
    """Print random parsed tables with their page, for hand-scoring."""
    docs = sorted(PARSED.glob("*.json"))[:files]
    for d in docs:
        data = json.loads(d.read_text(encoding="utf-8"))
        tables = [e for e in data["elements"] if e["kind"] == "table"]
        for t in random.sample(tables, min(n_per_file, len(tables))):
            print(f"\n### {data['source']} · page {t['page']} · rows={t['meta'].get('n_rows')}")
            print(t["text"][:600])
            print("SCORE [ ] 2=perfect 1=usable 0=broken   NOTE: ____")

if __name__ == "__main__":
    random.seed(20)
    sample()