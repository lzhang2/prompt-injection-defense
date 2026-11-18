"""Request and response shapes for the serving API (also drive the /docs page)."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

DefenseName = Literal["none", "spotlight", "classifier", "judge", "all"]


class EmailIn(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    sender: str = Field(max_length=320)
    subject: str = Field("", max_length=1000)
    body: str = Field(max_length=20000)


class AssistantRequest(BaseModel):
    request: str = Field(min_length=1, max_length=4000,
                         examples=["Summarize my most recent emails."])
    emails: Optional[List[EmailIn]] = Field(
        None, max_length=100,
        description="The user's inbox, oldest first. Omit to use the built-in demo inbox.")
    defense: Optional[DefenseName] = Field(
        None, description="Defense config. Omit to use serving.defense from the config.")


class EmailScore(BaseModel):
    id: str
    score: Optional[float] = Field(description="Injection probability; null if not scored")
    dropped: bool = Field(description="Removed before the LLM saw it")


class AssistantResponse(BaseModel):
    request_id: str
    answer: str
    tool_call: Optional[dict] = Field(description="What the LLM asked to do")
    tool_call_allowed: bool = Field(
        description="True only if there is a tool call and no defense blocked it. "
                    "The API never sends email itself; the caller acts on this.")
    blocked_by: Optional[str]
    emails: List[EmailScore] = Field(description="Retrieved emails and their scores")
    defense: str
    model_version: str


class ScoreRequest(BaseModel):
    texts: List[str] = Field(min_length=1, max_length=64)


class ScoreResult(BaseModel):
    score: float
    injection: bool


class ScoreResponse(BaseModel):
    request_id: str
    results: List[ScoreResult]
    threshold: float
    model_version: str
