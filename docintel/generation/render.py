from docintel.generation.schema import Claim


def render_gated(
    kept: list[Claim], rejected: list[Claim], chunk_lookup: dict[str, dict],
) -> str:
    """Kept claims render normally with resolved citations. Rejected
    claims render with a visible marker — the user sees that something
    was withheld, not a suspiciously short answer with no explanation."""
    lines = []
    for claim in kept:
        cites = "; ".join(
            f"{chunk_lookup[cid]['doc_id']} p.{chunk_lookup[cid]['page']}"
            for cid in claim.chunk_ids if cid in chunk_lookup
        ) or "no resolvable citation"
        lines.append(f"{claim.text} [{cites}]")
    for claim in rejected:
        lines.append(f'[insufficient support — claim withheld: "{claim.text[:60]}…"]')
    return "\n".join(lines)

if __name__ == "__main__":
    from docintel.gateway_client import GatewayClient
    from docintel.retrieval.pipeline import retrieve
    from docintel.generation.generate import generate_answer
    from docintel.generation.gate import faithfulness_gate

    client = GatewayClient()
    question = "What was R&D spend?"
    chunks, _ = retrieve(question, client, top_n=5, tenant_id="tenant_alpha")
    lookup = {c["chunk_id"]: c for c in chunks}
    answer = generate_answer(question, chunks, client)
    kept, rejected, stats = faithfulness_gate(answer, lookup, client)
    print(render_gated(kept, rejected, lookup))
    print(f"\n{stats}")