import pathlib
from sentence_transformers import SentenceTransformer
from dataclasses import asdict
from docintel.ingest.chunker import chunk_document
from docintel.ingest.embed import save_embeddings

# A strong open-source model — check MTEB for the latest leader.
# Note: BAAI/bge-base-en-v1.5 is also the tokenizer Day 21's HybridChunker
# was configured against, so this pairing is the "no re-chunk needed"
# baseline comparison. A genuinely different Model B would need its
# own tokenizer reconfigured back on Day 21 before chunk boundaries
# are trustworthy for it — see §1.5's migration-cost discussion.
MODEL_NAME = "BAAI/bge-base-en-v1.5"

def embed_local(chunks, model_name: str = MODEL_NAME) -> list[dict]:
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    # bge models use "Represent this sentence:" prefix for queries;
    # for passage encoding at ingest, no prefix needed (asymmetric!).
    vecs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    records = []
    for i, vec in enumerate(vecs):
        rec = asdict(chunks[i])
        rec["embedding"] = vec.tolist()
        rec["embed_model"] = model_name
        rec["embed_dims"] = len(vec)
        records.append(rec)
    return records

if __name__ == "__main__":
    for jp in sorted(pathlib.Path("parsed").glob("*.json")):
        cks = chunk_document(jp.stem, jp)
        recs = embed_local(cks)
        out = pathlib.Path("embeddings") / f"{jp.stem}.model_b.jsonl"
        save_embeddings(recs, out)
        print(f"{jp.stem}: {len(recs)} chunks, dims={recs[0]['embed_dims']}")