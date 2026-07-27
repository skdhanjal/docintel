from docintel.gateway_client import GatewayClient
from docintel.generation.schema import Claim

VERIFY_PROMPT = """Does the SOURCE passage support the CLAIM below? \
Answer with exactly one word: SUPPORTED or UNSUPPORTED. Do not \
explain. A claim is SUPPORTED only if the source directly states \
or clearly implies it — not if it merely relates to the same topic.

SOURCE:
{source}

CLAIM:
{claim}"""

def verify_claim(
    claim: Claim, chunk_lookup: dict[str, dict], client: GatewayClient,
) -> bool:
    """NLI-style entailment check. An unresolvable chunk_id (the model
    cited something not actually in its context) is an automatic
    failure — there is no source to verify against."""
    source = "\n\n".join(
        chunk_lookup[cid]["text"] for cid in claim.chunk_ids if cid in chunk_lookup
    )
    if not source:
        return False
    verdict = client.complete(
        VERIFY_PROMPT.format(source=source, claim=claim.text),
        model="cheap", max_tokens=5,
    )
    return verdict.strip().upper().startswith("SUPPORT")

if __name__ == "__main__":
    client = GatewayClient()
    lookup = {"c1": {"text": "R&D expense was $4.2 billion in fiscal 2023, up 11% year over year."}}
    supported = Claim(text="R&D spend was $4.2 billion in FY2023.", chunk_ids=["c1"])
    unsupported = Claim(text="R&D spend was $9.8 billion in FY2023.", chunk_ids=["c1"])
    print("supported claim  →", verify_claim(supported, lookup, client))
    print("unsupported claim →", verify_claim(unsupported, lookup, client))