import json
from collections import defaultdict
from docintel.gateway_client import GatewayClient
from docintel.retrieval.pipeline import retrieve
from docintel.eval.retrieval_metrics import score_query

def run_retrieval_eval(golden_path: str = "golden_set.json", k: int = 10, tenant_id: str = "tenant_alpha") -> dict:
    with open(golden_path) as f:
        golden = json.load(f)
    client = GatewayClient()
    per_archetype = defaultdict(list)
    all_scores = []

    for g in golden:
        print(f"Evaluating question {g['id']} ({g['archetype']})")
        results, _ = retrieve(g["question"], client, top_n=k, tenant_id=tenant_id, use_rerank=False)
        retrieved_ids = [r["chunk_id"] for r in results]
        scores = score_query(retrieved_ids, g["relevant_chunk_ids"], k=k)
        all_scores.append(scores)
        per_archetype[g["archetype"]].append(scores)

    def avg(scores_list, key):
        return sum(s[key] for s in scores_list) / len(scores_list)

    report = {"overall": {k: avg(all_scores, k) for k in ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"]}}
    report["by_archetype"] = {
        arch: {mk: avg(scores_list, mk) for mk in ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"]}
        for arch, scores_list in per_archetype.items()
    }
    return report

if __name__ == "__main__":
    report = run_retrieval_eval()
    print("Overall:", {k: round(v, 3) for k, v in report["overall"].items()})
    print("\nBy archetype:")
    for arch, scores in report["by_archetype"].items():
        print(f"  {arch:16s}", {k: round(v, 3) for k, v in scores.items()})