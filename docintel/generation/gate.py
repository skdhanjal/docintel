from docintel.gateway_client import GatewayClient
from docintel.generation.schema import Answer, Claim
from docintel.generation.verify import verify_claim

def faithfulness_gate(
    answer: Answer, chunk_lookup: dict[str, dict], client: GatewayClient,
) -> tuple[list[Claim], list[Claim], dict]:
    """Verify every claim independently; partition into kept/rejected.
    Nothing is silently dropped — the caller gets both lists plus stats."""
    kept, rejected = [], []
    for claim in answer.claims:
        if verify_claim(claim, chunk_lookup, client):
            kept.append(claim)
        else:
            rejected.append(claim)
    stats = {
        "total": len(answer.claims),
        "kept": len(kept),
        "rejected": len(rejected),
        "rejection_rate": len(rejected) / len(answer.claims) if answer.claims else 0.0,
    }
    return kept, rejected, stats