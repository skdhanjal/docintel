import json
from docintel.gateway_client import GatewayClient
from docintel.retrieval.pipeline import retrieve
from docintel.generation.generate import generate_answer
from docintel.eval.retrieval_metrics import score_query
from docintel.eval.judges import judge_faithfulness, citation_precision_recall

def run_full_eval(golden_path: str = "golden_set.json", tenant_id: str = "tenant_alpha") -> dict:
    with open(golden_path) as f:
        golden = json.load(f)
    client = GatewayClient()
    retrieval_scores, faithfulness_scores, citation_scores = [], [], []

    for g in golden:
        chunks, _ = retrieve(g["question"], client, top_n=10, tenant_id=tenant_id)
        retrieval_scores.append(score_query([c["chunk_id"] for c in chunks], g["relevant_chunk_ids"]))

        answer = generate_answer(g["question"], chunks[:5], client)
        cited = [cid for claim in answer.claims for cid in claim.chunk_ids]
        citation_scores.append(citation_precision_recall(cited, g["relevant_chunk_ids"]))

        source = "\n\n".join(c["text"] for c in chunks[:5])
        answer_text = " ".join(claim.text for claim in answer.claims)
        fj = judge_faithfulness(answer_text, source, client)
        if fj.get("score") is not None:
            faithfulness_scores.append(fj["score"])

    def avg(items, key):
        return round(sum(i[key] for i in items) / len(items), 3) if items else 0.0

    return {
        "retrieval": {mk: avg(retrieval_scores, mk) for mk in ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"]},
        "generation": {
            "faithfulness_avg": round(sum(faithfulness_scores) / len(faithfulness_scores), 3) if faithfulness_scores else 0.0,
            "citation_precision": avg(citation_scores, "precision"),
            "citation_recall": avg(citation_scores, "recall"),
        },
    }

if __name__ == "__main__":
    report = run_full_eval()
    print("=== RETRIEVAL ===")
    print(json.dumps(report["retrieval"], indent=2))
    print("\n=== GENERATION ===")
    print(json.dumps(report["generation"], indent=2))