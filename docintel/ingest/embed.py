import json, pathlib, time
from dataclasses import asdict
from docintel.gateway_client import GatewayClient
from docintel.ingest.chunker import Chunk, chunk_document

BATCH_SIZE = 64   # most API limits are 2048 inputs; 64 is safe and fast

def embed_chunks(
    chunks: list[Chunk],
    client: GatewayClient,
    model: str = "text-embedding-3-small",
) -> list[dict]:
    """Return list of {chunk dict + 'embedding' + 'embed_model'}.
    chunk.text is already chunker.contextualize()'s output from Day 21 --
    headings woven in for both prose and tables. Nothing further to
    enrich here; that semantic-bridge work is done at chunk time."""
    records = []
    texts = [c.text for c in chunks]

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        vecs  = client.embed(batch, model=model)
        for j, vec in enumerate(vecs):
            rec = asdict(chunks[i+j])
            rec["embedding"] = vec
            rec["embed_model"] = model
            rec["embed_dims"] = len(vec)
            records.append(rec)
        time.sleep(0.1)  # gentle on rate limits
    return records

def save_embeddings(records: list[dict], out: pathlib.Path):
    """Persist to JSONL — one embedded chunk per line."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"saved {len(records)} records to {out}")

if __name__ == "__main__":
    client = GatewayClient()
    assert client.health(), "start llm-gateway first"

    for jp in sorted(pathlib.Path("parsed").glob("*.json")):
        # Reload the cached DoclingDocument and chunk it fresh (Day 21) --
        # no need to read data/chunked here, chunk_document() is fast
        # and this keeps the embedding step self-contained.
        cks = chunk_document(jp.stem, jp)
        recs = embed_chunks(cks, client, model="text-embedding-3-small")
        out = pathlib.Path("embeddings") / f"{jp.stem}.model_a.jsonl"
        out.parent.mkdir(exist_ok=True)
        save_embeddings(recs, out)
        print(f"{jp.stem}: {len(recs)} chunks, dims={recs[0]['embed_dims']}")