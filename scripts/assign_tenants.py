import psycopg

CONN_STR = "postgresql://docintel:docintel_dev@localhost/docintel"

# Adjust tickers to whichever five filings you loaded on Day 19.
# Split deliberately uneven — it's a more realistic test than 50/50.
TENANT_MAP = {
    "CVX": "tenant_alpha", "JPM": "tenant_alpha", "NVDA": "tenant_alpha",
    "WMT":  "tenant_beta",  "PFE":  "tenant_beta",
}

def assign_tenants():
    conn = psycopg.connect(CONN_STR)
    for ticker, tenant in TENANT_MAP.items():
        conn.execute(
            "UPDATE chunks SET tenant_id = %s WHERE doc_id LIKE %s",
            (tenant, f"{ticker}%"),
        )
    conn.commit()
    rows = conn.execute(
        "SELECT tenant_id, count(*) FROM chunks GROUP BY tenant_id ORDER BY 1"
    ).fetchall()
    for t, n in rows:
        print(f"  {t}: {n} chunks")
    conn.close()

if __name__ == "__main__":
    assign_tenants()