import psycopg

CONN_STR = "postgresql://docintel:docintel_dev@localhost/docintel"

def add_access_control():
    conn = psycopg.connect(CONN_STR)

    # 1. Role-level ACL, orthogonal to tenant_id: which groups within
    #    the tenant may see this chunk (e.g. "legal", "exec", "all").
    conn.execute("""
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS
            acl_groups text[] NOT NULL DEFAULT '{}'
    """)

    # 2. GIN index for array-overlap filters (acl_groups && %s).
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_acl
            ON chunks USING gin (acl_groups)
    """)

    # 3. Backfill: existing chunks get an "all-access" group so
    #    nothing silently disappears from prior days' queries.
    conn.execute("""
        UPDATE chunks SET acl_groups = ARRAY['all']
        WHERE acl_groups = '{}'
    """)

    # 4. Defense in depth: Row-Level Security. FORCE means even the
    #    table owner is subject to the policy (superusers still bypass
    #    RLS by default — the app connects as a non-superuser role).
    conn.execute("ALTER TABLE chunks ENABLE ROW LEVEL SECURITY")
    conn.execute("ALTER TABLE chunks FORCE ROW LEVEL SECURITY")
    conn.execute("DROP POLICY IF EXISTS tenant_isolation ON chunks")
    conn.execute("""
        CREATE POLICY tenant_isolation ON chunks
            USING (tenant_id = current_setting('app.tenant_id', true))
    """)

    conn.commit()
    print("✓ acl_groups column + GIN index + RLS policy created")
    conn.close()

if __name__ == "__main__":
    add_access_control()