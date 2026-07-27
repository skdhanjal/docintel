import time
from docintel.gateway_client import GatewayClient
from docintel.retrieval.pipeline import retrieve
from docintel.generation.generate import generate_answer
from docintel.generation.gate import faithfulness_gate

QUESTIONS = [
    "What was total revenue in fiscal 2023?",
    "What was R&D spend?",
    "Describe the debt maturity schedule.",
]

client = GatewayClient()
total_gen_ms, total_gate_ms, total_claims, total_rejected = 0.0, 0.0, 0, 0

for q in QUESTIONS:
    chunks, _ = retrieve(q, client, top_n=5, tenant_id="tenant_alpha")
    lookup = {c["chunk_id"]: c for c in chunks}

    t0 = time.perf_counter()
    answer = generate_answer(q, chunks, client)
    t1 = time.perf_counter()
    kept, rejected, stats = faithfulness_gate(answer, lookup, client)
    t2 = time.perf_counter()

    gen_ms, gate_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
    total_gen_ms += gen_ms; total_gate_ms += gate_ms
    total_claims += stats["total"]; total_rejected += stats["rejected"]

    print(f"{q[:40]:40s}  gen={gen_ms:5.0f}ms  gate={gate_ms:5.0f}ms  "
          f"claims={stats['total']}  rejected={stats['rejected']}")

print(f"\nTotal gen: {total_gen_ms:.0f}ms   Total gate: {total_gate_ms:.0f}ms")
print(f"Gate overhead: {total_gate_ms / total_gen_ms:.1f}x generation cost")
print(f"Overall rejection rate: {total_rejected}/{total_claims} claims")
print("\nRecord both numbers in DECISIONS.md.")