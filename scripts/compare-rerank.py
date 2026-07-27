from docintel.gateway_client import GatewayClient
from docintel.retrieval.pipeline import retrieve

QUERIES = [
    "What was total revenue in fiscal 2023?",
    "What are the main risk factors?",
    "What was R&D spend?",
    "Describe the debt maturity schedule.",
    "Item 7A market risk disclosures",
]

client = GatewayClient()
totals = {"on": [], "off": []}

print(f"{'Query':42s}  {'rerank=OFF':28s}  {'rerank=ON':28s}")
print("="*102)

for q in QUERIES:
    off_results, off_t = retrieve(q, client, top_n=5, use_rerank=False)
    on_results, on_t = retrieve(q, client, top_n=5, use_rerank=True)
    totals["off"].append(off_t["total_ms"])
    totals["on"].append(on_t["total_ms"])

    off_top = off_results[0]["text"][:26] if off_results else ""
    on_top = on_results[0]["text"][:26] if on_results else ""
    print(f"{q[:40]:40s}  {off_t['total_ms']:5.0f}ms  {off_top:20s}  "
          f"{on_t['total_ms']:5.0f}ms  {on_top:20s}")

avg_off = sum(totals["off"]) / len(totals["off"])
avg_on = sum(totals["on"]) / len(totals["on"])
print(f"\nAvg total latency  rerank=OFF: {avg_off:.0f}ms   rerank=ON: {avg_on:.0f}ms")
print(f"Delta: +{avg_on - avg_off:.0f}ms for reranking")
print("\nManually check: did the #1 result change for the better on any query?")
print("Record both configs' top-5 order + timing in DECISIONS.md.")