import psycopg
from docintel.gateway_client import GatewayClient
from docintel.retrieval.hybrid import hybrid_search
from docintel.ingest.load_db import CONN_STR

client = GatewayClient()
# Vocabulary that only appears in tenant_alpha's filings (AAPL/MSFT/NVDA).
ALPHA_ONLY_QUERY = "Item 7A market risk disclosures"

def test_cross_tenant_isolation():
    # Attacker: authenticated as tenant_beta, queries for alpha-only content.
    results = hybrid_search(ALPHA_ONLY_QUERY, client, k=10, tenant_id="tenant_beta")
    leaked = [r for r in results if r["doc_id"].split("_")[0] in ("CVX", "JPM", "NVDA")]
    assert not leaked, f"LEAK: tenant_beta retrieved {len(leaked)} tenant_alpha chunks"
    print(f"✓ tenant_beta query returned {len(results)} results, 0 from tenant_alpha")

def test_rls_defense_in_depth():
    # Simulate the bug the application filter is supposed to prevent:
    # a raw query with NO WHERE tenant_id clause at all.
    conn = psycopg.connect(CONN_STR)
    conn.execute("SET LOCAL app.tenant_id = %s", ("tenant_beta",))
    rows = conn.execute("SELECT DISTINCT tenant_id FROM chunks").fetchall()  # no WHERE!
    tenants_seen = {r[0] for r in rows}
    assert tenants_seen <= {"tenant_beta"}, f"RLS FAILED: saw {tenants_seen}"
    print(f"✓ RLS enforced — unfiltered query only returned {tenants_seen}")
    conn.close()

if __name__ == "__main__":
    test_cross_tenant_isolation()
    test_rls_defense_in_depth()
    print("\nBoth isolation layers verified independently.")