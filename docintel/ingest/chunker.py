from __future__ import annotations
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
import pathlib
from dataclasses import dataclass, field

from docintel.ingest.parse_pipeline import load_cached

# Must match the embedding model you call on Day 22 -- the chunker's
# token budget is only meaningful if it's measured against the same
# tokenizer the embedding model actually uses.
EMBED_MODEL_ID = "BAAI/bge-base-en-v1.5"
MAX_TOKENS = 500   # bge-base's real context limit -- not a guess

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,   # merge undersized peer chunks sharing the same heading path
)

@dataclass
class Chunk:
    chunk_id: str
    text: str                        # chunker.contextualize(chunk) -- what gets embedded
    kind: str                        # "text" | "table" (from doc_items[0].label)
    # ---- metadata, all read from chunk.meta, nothing hand-derived ----
    doc_id: str           = ""
    page: int | None      = None
    headings: list[str] = field(default_factory=list)   # ["Item 7A", "Market Risk"]
    token_count: int      = 0

def _uid(doc_id: str, index: int) -> str:
    return f"{doc_id}_c{index:04d}"

def chunk_document(doc_id: str, json_path: pathlib.Path) -> list[Chunk]:
    """Reload the cached DoclingDocument (no PDF, no layout model --
    Day 20's cache boundary paying off) and run HybridChunker over it."""
    doc = load_cached(json_path)
    out: list[Chunk] = []

    for i, raw in enumerate(chunker.chunk(dl_doc=doc)):
        # contextualize() is what actually gets embedded on Day 22 --
        # it prepends headings (and, for tables, caption/context) to
        # the raw text, which is exactly the semantic-bridge fix a
        # hand-rolled "table NL description" step would otherwise
        # exist to provide. HybridChunker gives it to you for free.
        embed_text = chunker.contextualize(chunk=raw)

        item = raw.meta.doc_items[0] if raw.meta.doc_items else None
        kind = item.label if item else "text"
        page = item.prov[0].page_no if item and item.prov else None

        out.append(Chunk(
            chunk_id=_uid(doc_id, i),
            text=embed_text,
            kind="table" if kind == "table" else "text",
            doc_id=doc_id,
            page=page,
            headings=list(raw.meta.headings or []),
            token_count=tokenizer.count_tokens(embed_text),
        ))
    return out

if __name__ == "__main__":
    import pathlib
    from collections import Counter
    
    p = next(pathlib.Path("parsed").glob("*.json"))
    print(f"chunking {p.name}...")
    cks = chunk_document(p.stem, p)
    print("chunks:", len(cks))
    print(Counter(c.kind for c in cks))
    tbl = next((c for c in cks if c.kind == "table"), None)
    
    if tbl:
        print(f"\nsample table chunk (page {tbl.page}, headings {tbl.headings}, {tbl.token_count} tokens):")
        print(tbl.text)