import hashlib
from collections import defaultdict
from .chunker import Chunk

def parent_key(doc_id: str, headings: list[str]) -> str:
    """Stable, deterministic key for a heading path -- a short hash,
    not a random UUID. Any chunk can independently recompute its own
    parent_key from (doc_id, headings) alone, with no lookup needed.
    This is what makes it a real, storable, joinable column (Day 23)
    instead of an in-memory-only grouping."""
    raw = f"{doc_id}::{'/'.join(headings)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]

def build_parent_map(chunks: list[Chunk]) -> dict[str, str]:
    """parent_key -> concatenated text of every leaf sharing that
    heading path. The generator gets this instead of one leaf, so
    it reads a coherent section -- heading, prose, table -- together."""
    groups: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        groups[parent_key(c.doc_id, c.headings)].append(c.text)
    return {key: "\n\n".join(parts) for key, parts in groups.items()}

def parent_for(chunk: Chunk, pmap: dict[str, str]) -> str:
    """Return the parent text a leaf belongs to, or the leaf itself
    if it has no headings (e.g. content before the first heading)."""
    return pmap.get(parent_key(chunk.doc_id, chunk.headings), chunk.text)

if __name__ == "__main__":
    import pathlib
    from .chunker import chunk_document
    p = next(pathlib.Path("parsed").glob("*.json"))
    cks = chunk_document(p.stem, p)
    pmap = build_parent_map(cks)
    print(f"parents: {len(pmap)},  avg parent size: "
          f"{sum(len(v) for v in pmap.values())//len(pmap)} chars")