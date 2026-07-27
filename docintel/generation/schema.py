from pydantic import BaseModel

class Claim(BaseModel):
    text: str
    chunk_ids: list[str]

class Answer(BaseModel):
    claims: list[Claim]