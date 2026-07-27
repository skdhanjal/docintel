import psycopg

DDL = """
-- Chunks: the retrieval atoms. Fields mirror Day 21's Chunk dataclass
-- plus parent_key, computed at load time from (doc_id, headings).
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    kind        TEXT NOT NULL,                       -- 'text' | 'table' (doc_items[0].label)
    page        INTEGER,
    headings    TEXT[],                               -- chunk.meta.headings, breadcrumb citations
    parent_key  TEXT,                                 -- → parent_docs(parent_key)
    text        TEXT NOT NULL,                        -- chunker.contextualize() output, what was embedded
    token_count INTEGER,                              -- computed once at chunk time (Day 21)
    embed_model TEXT NOT NULL,
    embedding   vector({dims}),                       -- pgvector type
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Parent docs: the section-level context for parent-document retrieval.
-- Keyed identically to chunks.parent_key -- same hash function,
-- computed from the same (doc_id, headings) pair (Day 21 §2.3).
CREATE TABLE IF NOT EXISTS parent_docs (
    parent_key  TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    text        TEXT NOT NULL
);

-- Indexes.
-- 1. HNSW for ANN search — cosine distance.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = {m}, ef_construction = {ef_construction});

-- 2. B-tree on tenant_id for mandatory filters.
CREATE INDEX IF NOT EXISTS idx_chunks_tenant
    ON chunks (tenant_id);

-- 3. B-tree on doc_id for per-document queries.
CREATE INDEX IF NOT EXISTS idx_chunks_doc
    ON chunks (doc_id);

-- 4. B-tree on kind for type-filtered retrieval.
CREATE INDEX IF NOT EXISTS idx_chunks_kind
    ON chunks (kind);

-- 5. B-tree on parent_key -- the join Day 21 promised would be "one lookup."
CREATE INDEX IF NOT EXISTS idx_chunks_parent
    ON chunks (parent_key);
"""

def create(dims: int = 1536, m: int = 16, ef_construction: int = 128):
    conn = psycopg.connect("postgresql://docintel:docintel_dev@localhost/docintel")
    conn.execute(DDL.format(dims=dims, m=m, ef_construction=ef_construction))
    conn.commit()
    print(f"schema created: dims={dims}, m={m}, ef_c={ef_construction}")
    conn.close()

if __name__ == "__main__":
    create()