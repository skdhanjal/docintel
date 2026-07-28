import json, random
from docintel.gateway_client import GatewayClient
from docintel.retrieval.pipeline import retrieve
from docintel.generation.generate import generate_answer
from docintel.generation.render import render_with_citations
from docintel.eval.judges import judge_faithfulness

def sample_for_labeling(n: int = 30) -> list[dict]:
    with open("golden_set.json") as f:
        golden = json.load(f)
    client = GatewayClient()
    sample = random.sample(golden, min(n, len(golden)))
    examples = []
    for g in sample:
        chunks, _ = retrieve(g["question"], client, top_n=5, tenant_id="tenant_alpha")
        lookup = {c["chunk_id"]: c for c in chunks}
        answer = generate_answer(g["question"], chunks, client)
        rendered = render_with_citations(answer, lookup)
        source = "\n\n".join(c["text"] for c in chunks)
        examples.append({"id": g["id"], "question": g["question"], "answer": rendered, "source": source, "human_score": None})
    with open("calibration_set.json", "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Wrote {len(examples)} examples to calibration_set.json — fill in human_score (0/1/2) for each, then run --score")

def score_agreement():
    with open("calibration_set.json") as f:
        examples = json.load(f)
    assert all(e["human_score"] is not None for e in examples), "label every example first"
    client = GatewayClient()
    exact, within_one, total = 0, 0, len(examples)
    for e in examples:
        j = judge_faithfulness(e["answer"], e["source"], client)
        judge_score = j.get("score")
        if judge_score is None:
            continue
        if judge_score == e["human_score"]:
            exact += 1
        if abs(judge_score - e["human_score"]) <= 1:
            within_one += 1
    print(f"Exact agreement:      {exact}/{total} ({exact/total:.0%})")
    print(f"Within-one agreement: {within_one}/{total} ({within_one/total:.0%})")
    print("Record both numbers in DECISIONS.md — this is your judge's calibration.")

if __name__ == "__main__":
    import sys
    if "--score" in sys.argv:
        score_agreement()
    else:
        sample_for_labeling()