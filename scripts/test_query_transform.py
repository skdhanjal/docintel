from docintel.gateway_client import GatewayClient
from docintel.retrieval.pipeline import retrieve
from docintel.retrieval.query_transform import decompose_retrieve

client = GatewayClient()
question = "Compare R&D spending trends across the three tech filings."

# Direct: one query, straight through the Day 26 pipeline.
direct, _ = retrieve(question, client, top_n=5)

# Decomposed: one sub-query per company, concatenated.
transformed, sub_qs = decompose_retrieve(question, client, top_n_per_sub=3)

def doc_coverage(results):
    return sorted({r["doc_id"] for r in results})

print(f"Question: {question}\n")
print("— Direct query —")
print(f"Docs covered: {doc_coverage(direct)}")
for r in direct:
    print(f"  {r['doc_id'][:16]:16s}  {r['text'][:55]}")

print("\n— Decomposed —")
print("Sub-questions:")
for sq in sub_qs:
    print(f"  - {sq}")
print(f"Docs covered: {doc_coverage(transformed)}")
for r in transformed:
    print(f"  {r['doc_id'][:16]:16s}  {r['text'][:55]}")

print("\nRecord doc-coverage counts for both in DECISIONS.md.")