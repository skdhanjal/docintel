import psycopg

CONN_STR = "postgresql://docintel:docintel_dev@localhost/docintel"

def add_fulltext():
    conn = psycopg.connect(CONN_STR)

    # 1. Add tsvector column if not exists.
    conn.execute("""
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS
            tsv tsvector
    """)

    # 2. Populate from the text column (english config).
    conn.execute("""
        UPDATE chunks SET tsv = to_tsvector('english', text)
        WHERE tsv IS NULL
    """)

    # 3. GIN index for fast full-text search.
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_fts
            ON chunks USING gin (tsv)
    """)

    # 4. Auto-update trigger: new inserts get tsv automatically.
    conn.execute("""
        CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.tsv := to_tsvector('english', COALESCE(NEW.text, ''));
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS trg_chunks_tsv ON chunks;
        CREATE TRIGGER trg_chunks_tsv
            BEFORE INSERT OR UPDATE ON chunks
            FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();
    """)

    conn.commit()
    # Verify
    n = conn.execute("SELECT count(*) FROM chunks WHERE tsv IS NOT NULL").fetchone()[0]
    print(f"✓ tsvector populated on {n} chunks, GIN index + trigger created")
    conn.close()

if __name__ == "__main__":
    add_fulltext()