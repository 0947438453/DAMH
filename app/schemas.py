from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    question: str
    source: Optional[str] = "auto"  # "local" | "web" | "both" | "auto"

class ChatResponse(BaseModel):
    answer: str
    used_sources: List[str]
