import json
from docintel.gateway_client import GatewayClient

FAITHFULNESS_RUBRIC = """Score how faithful the ANSWER is to the SOURCE passages on a \
scale of 0, 1, or 2 — do not use any other values:

  0 = Contains at least one claim that contradicts or is absent \
      from the source (a fabricated or wrong figure, an unsupported \
      assertion). An approximate figure that rounds correctly \
      (e.g. "around $4B" when the source says $4.2B) is NOT a \
      violation. An exact-but-wrong figure IS a violation.
  1 = All claims are supported, but the answer omits source \
      context that materially changes interpretation (e.g. stating \
      a growth number without noting it was driven by a one-time item \
      the source explicitly flags).
  2 = Fully faithful: every claim is directly supported and no \
      materially relevant context is omitted.

Respond with JSON only: {{"score": 0|1|2, "reason": "one sentence"}}

SOURCE:
{source}

ANSWER:
{answer}"""

def judge_faithfulness(answer_text: str, source_text: str, client: GatewayClient) -> dict:
    raw = client.complete(
        FAITHFULNESS_RUBRIC.format(source=source_text, answer=answer_text),
        model="cheap", max_tokens=120,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": None, "reason": f"judge returned non-JSON: {raw[:80]}"}
    
def citation_precision_recall(cited_chunk_ids: list[str], relevant_chunk_ids: list[str]) -> dict:
    """Of the chunks the GENERATOR cited, how many are actually
    relevant to the question (precision)? Of the relevant chunks,
    how many did the generator actually cite (recall)? Distinct
    from retrieval recall — this measures what made it into the
    final answer's citations, not what retrieval fetched."""
    cited = set(cited_chunk_ids)
    relevant = set(relevant_chunk_ids)
    if not cited and not relevant:
        return {"precision": 1.0, "recall": 1.0}
    hit = len(cited & relevant)
    precision = hit / len(cited) if cited else 0.0
    recall = hit / len(relevant) if relevant else 0.0
    return {"precision": precision, "recall": recall}
    

if __name__ == "__main__":
    client = GatewayClient()
    source = "R&D expense was $4.2 billion in fiscal 2023, up 11% year over year, driven primarily by AI infrastructure investment."
    faithful = "R&D spending was $4.2 billion in FY2023, an 11% increase, mainly due to AI infrastructure investment."
    fabricated = "R&D spending was $9.8 billion in FY2023, more than double the prior year."
    print("faithful  →", judge_faithfulness(faithful, source, client))
    print("fabricated →", judge_faithfulness(fabricated, source, client))
    
    # (appended to the __main__ block from §2.1)
    print("citation check →", citation_precision_recall(
        cited_chunk_ids=["c1", "c2"],
        relevant_chunk_ids=["c1", "c3"],
    ))  # precision 0.5 (c2 shouldn't have been cited), recall 0.5 (c3 was missed)