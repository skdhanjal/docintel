import math

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hit = len(set(retrieved_ids[:k]) & relevant_ids)
    return hit / len(relevant_ids)

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    hit = len(set(retrieved_ids[:k]) & relevant_ids)
    return hit / k if k else 0.0

def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary relevance nDCG: DCG@k / IDCG@k, log2-rank-discounted."""
    dcg = sum(
        1.0 / math.log2(i + 2) for i, cid in enumerate(retrieved_ids[:k]) if cid in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0

def score_query(retrieved_ids: list[str], relevant_ids: list[str], k: int = 10) -> dict:
    rel = set(relevant_ids)
    return {
        "recall_at_k": recall_at_k(retrieved_ids, rel, k),
        "precision_at_k": precision_at_k(retrieved_ids, rel, k),
        "mrr": mrr(retrieved_ids, rel),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, rel, k),
    }