import pathlib

from docintel.ingest.parse_pipeline import parse_and_cache

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/parsed"); OUT.mkdir(parents=True, exist_ok=True)

for pdf in sorted(RAW.glob("*.pdf")):
    parse_and_cache(pdf, OUT)