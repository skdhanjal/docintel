import json
from docintel.gateway_client import GatewayClient
from docintel.generation.schema import Answer

SYSTEM_PROMPT = """You are a financial analyst assistant. Answer the \
user's question using ONLY the numbered context chunks provided. \
Break your answer into discrete factual claims — each claim should \
be a single, checkable statement. Every claim MUST cite the \
chunk_id(s) it is directly based on. If the context does not \
support a claim, do not make it; it is correct to produce fewer \
claims rather than an unsupported one.

Respond with JSON matching this shape exactly, no other text:
{"claims": [{"text": "...", "chunk_ids": ["chunk_id_1", ...]}, ...]}
"""

def build_context_block(chunks: list[dict]) -> str:
    """Each chunk is labeled with the exact chunk_id the model must
    echo back in citations — no separate ID-resolution step needed."""
    lines = []
    for c in chunks:
        lines.append(f"[{c['chunk_id']}] (doc={c['doc_id']}, p.{c['page']})\n{c['text']}")
    return "\n\n".join(lines)

def generate_answer(question: str, chunks: list[dict], client: GatewayClient) -> Answer:
    context = build_context_block(chunks)
    messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    resp = client.generate(
        SYSTEM_PROMPT, messages, model="gpt-4o-mini",
        response_format={"type": "json_object"},
    )
    raw = resp["content"][0]["text"] if isinstance(resp.get("content"), list) else resp["text"]
    return Answer.model_validate(json.loads(raw))

if __name__ == "__main__":
    from docintel.retrieval.pipeline import retrieve
    client = GatewayClient()
    question = "What was R&D spend?"
    chunks, timings = retrieve(question, client, top_n=5, tenant_id="tenant_alpha", use_rerank=False)
    answer = generate_answer(question, chunks, client)
    print(f"Retrieval timings: {timings}")
    for claim in answer.claims:
        print(f"  {claim.text}  [{', '.join(claim.chunk_ids)}]")