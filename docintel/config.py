from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # docintel never calls a provider directly — only the gateway.
    gateway_url: str = "http://localhost:8000"
    corpus_dir: str = "corpus"
    # SEC requires a descriptive UA with contact info (fair-access policy).
    sec_user_agent: str = "docintel-study rag@docintel.com"

    class Config:
        env_file = ".env"

settings = Settings()