import json, pathlib
from docintel.ingest.parse_pipeline import content_hash

MANIFEST_PATH = pathlib.Path("manifest.json")

def load_manifest() -> dict[str, str]:
    """filename → content_hash of everything previously ingested."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}

def save_manifest(m: dict[str, str]):
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))

def scan_corpus(corpus_dir: pathlib.Path) -> dict[str, str]:
    """filename → content_hash of everything currently on disk."""
    out = {}
    for f in sorted(corpus_dir.iterdir()):
        if f.suffix.lower() in {".pdf", ".html", ".docx"}:
            out[f.name] = content_hash(f)
    return out

if __name__ == "__main__":
    from docintel.config import settings
    current = scan_corpus(pathlib.Path(settings.corpus_dir))
    print("current corpus:")
    for name, h in current.items():
        print(f"  {name}: {h}")