from fastapi import FastAPI, HTTPException

from . import advisor, config, rag
from .schemas import AskRequest, AskResponse, Profile, RecommendResponse

app = FastAPI(title="LIC Policy Advisor (RAG)", version="0.1.0")


def _require_index():
    if rag.indexed_count() == 0:
        raise HTTPException(503, "Index is empty. Put brochure PDFs in data/raw and run: python -m scripts.ingest")


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": rag.indexed_count()}


@app.get("/plans")
def plans():
    df, skipped = advisor.clean_policies(advisor.load_policies())
    return {"complete": df.plan.tolist(), "incomplete": skipped}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    _require_index()
    try:
        result = rag.answer_question(body.question, plan=body.plan, top_k=body.top_k)
    except Exception as exc:  # LLM/auth/network failures
        raise HTTPException(502, f"LLM call failed: {type(exc).__name__}") from exc
    return {**result, "disclaimer": config.DISCLAIMER}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(profile: Profile):
    _require_index()
    try:
        return advisor.recommend(profile.model_dump(exclude={"top_n"}), top_n=profile.top_n)
    except Exception as exc:
        raise HTTPException(502, f"Recommendation failed: {type(exc).__name__}") from exc
