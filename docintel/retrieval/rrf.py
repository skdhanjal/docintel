from collections import defaultdict

K_CONSTANT = 60  # RRF smoothing constant (standard default)

def reciprocal_rank_fusion(
    *ranked_lists: list[list[str]],
    k: int = K_CONSTANT,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    Each input is a list of chunk_ids in rank order (index 0 = rank 1).
    Returns [(chunk_id, rrf_score)] sorted by score descending.

    RRF score for document d:
        score(d) = Σ  1 / (k + rank_i(d))
    where the sum is over all rankers that include d.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank_0, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank_0 + 1)  # rank is 1-based
    return sorted(scores.items(), key=lambda x: -x[1])

if __name__ == "__main__":
    # Self-test: chunk_A appears in both lists, chunk_E only in sparse.
    dense  = ["A", "C", "F", "B", "D"]
    sparse = ["E", "A", "G", "B", "H"]
    fused = reciprocal_rank_fusion(dense, sparse)
    print("RRF self-test:")
    for cid, score in fused[:6]:
        print(f"  {cid}: {score:.4f}")
    assert fused[0][0] == "A", "A should rank #1 (in both lists)"
    assert "E" in [c for c, _ in fused[:5]], "E should appear in top 5 (sparse #1)"
    print("✓ RRF self-test passed")