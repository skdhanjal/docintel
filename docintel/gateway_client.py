import httpx
from docintel.config import settings

class GatewayClient:
    """All docintel inference goes through here. Kept deliberately
    minimal today; embed() and generate() land on Days 22 and 29."""
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.gateway_url
        self._c = httpx.Client(timeout=60.0)

    def health(self) -> bool:
        try:
            return self._c.get(f"{self.base_url}/docs").status_code == 200
        except httpx.HTTPError:
            return False
        
    def embed(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        """Batch-embed via the gateway (which proxies to the provider).
        Returns one vector per input text."""
        
        r = self._c.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": texts, "model": model},
            timeout=120.0,
        )
        r.raise_for_status()
        
        return [d["embedding"] for d in r.json()["data"]]   
    
    def complete(self, prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 300) -> str:
        """Single-turn completion via the gateway's cheap-model route.

        Used for query transformation (rewriting, expansion, decomposition,
        HyDE) — not final answer generation, which is generate() on Day 29
        and always uses the citation-required, verified path.
        """
        r = self._c.post(f"{self.base_url}/v1/complete", json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        })
        r.raise_for_status()
        return r.json()["text"].strip() 
    
    def generate(
        self,
        system: str,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        max_tokens: int = 900,
        response_format: dict | None = None,
    ) -> dict:
        """Full generation — the citation-required path for user-facing
        answers. Distinct from complete(): this is where DECISIONS.md's
        "generation" model tier is spent, not the cheap-transformation tier.
        """
        payload = {
            "model": model, "system": system,
            "messages": messages, "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        r = self._c.post(f"{self.base_url}/v1/messages", json=payload)
        r.raise_for_status()
        return r.json()

if __name__ == "__main__":
    ok = GatewayClient().health()
    print("gateway reachable:", ok)
    assert ok, "start llm-gateway first: uvicorn gateway.main:app"