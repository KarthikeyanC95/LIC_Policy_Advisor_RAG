from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=["What is paid on death during the policy term?"])
    plan: str | None = Field(default=None, description="Restrict the search to one brochure (file name without .pdf)")
    top_k: int = Field(default=4, ge=1, le=10)


class Source(BaseModel):
    plan: str
    page: int
    similarity: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    disclaimer: str


class Profile(BaseModel):
    age: int = Field(ge=0, le=100)
    goal: str = Field(min_length=2, examples=["protection"])
    term: int = Field(gt=0, le=100, description="Policy term in years")
    sum_assured: float = Field(gt=0, description="Rupees")
    annual_budget: float | None = Field(default=None, gt=0, description="Rupees per year")
    top_n: int = Field(default=3, ge=1, le=10)


class PlanResult(BaseModel):
    plan: str
    benefits_summary: str
    estimated_annual_premium: float | None
    premium_note: str
    within_budget: bool | None


class RecommendResponse(BaseModel):
    plans: list[PlanResult]
    skipped_incomplete_rows: list[str]
    disclaimer: str
