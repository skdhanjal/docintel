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

if __name__ == "__main__":
    ok = GatewayClient().health()
    print("gateway reachable:", ok)
    assert ok, "start llm-gateway first: uvicorn gateway.main:app"