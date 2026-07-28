from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Loaded once per process and reused — a CrossEncoder load is
# expensive (model weights), scoring a batch of pairs is cheap.
_model: CrossEncoder | None = None

def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME, max_length=512)
    return _model

def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """Re-score candidates with a cross-encoder; return the top_n.

    Each candidate must have a "text" key (chunk_id, metadata etc.
    are preserved and passed through). Adds a "rerank_score" key.
    """
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)  # one batched forward pass
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: -c["rerank_score"])
    return ranked[:top_n]

if __name__ == "__main__":
    # Self-test: an obviously-relevant passage should outrank
    # an obviously-irrelevant one for a simple query.
    query = "What was R&D spend in fiscal 2023?"
    candidates = [
        {"chunk_id": "c1", "text": "Research and development expenses were $4.2 billion in fiscal 2023, up 11% year over year."},
        {"chunk_id": "c2", "text": "The company's headquarters lease was renewed for an additional ten years."},
        {"chunk_id": "c3", "text": "Marketing spend increased due to a new product launch campaign."},
    ]
    top = rerank(query, candidates, top_n=3)
    print("Rerank self-test:")
    for c in top:
        print(f"  {c['rerank_score']:.3f}  {c['chunk_id']}  {c['text'][:60]}")
    assert top[0]["chunk_id"] == "c1", "the R&D passage should rank #1"
    print("✓ rerank self-test passed")