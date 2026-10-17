"""Retrieval + generation. Heavy imports are lazy so the API starts fast and tests stay light."""
from functools import lru_cache

from . import config

SYSTEM_PROMPT = """You are PolicyBuddy, an assistant that explains LIC policy brochures.
Use ONLY the provided context.
Rules:
- Cite every fact as [plan name, p.X].
- If the context does not contain the answer, say "This is not in the brochures I have" and do not guess.
- NEVER invent or calculate premium amounts, bonus rates or returns. Quote only figures that appear in the context.
- Keep answers simple. This is general information, not financial advice."""


@lru_cache
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache
def collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection("lic_kb", metadata={"hnsw:space": "cosine"})


@lru_cache
def _llm():
    from groq import Groq
    return Groq()  # reads GROQ_API_KEY from the environment


def embed_texts(texts):
    return _embedder().encode(texts, normalize_embeddings=True).tolist()


def indexed_count() -> int:
    return collection().count()


def retrieve(query: str, top_k: int = 4, plan: str | None = None):
    kwargs = {"where": {"plan": plan}} if plan else {}
    res = collection().query(query_embeddings=embed_texts([query]), n_results=top_k, **kwargs)
    return [
        {"text": t, "plan": m["plan"], "page": m["page"], "similarity": 1 - d}
        for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def ask_llm(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    r = _llm().chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return r.choices[0].message.content


def build_context(chunks) -> str:
    return "\n\n".join(f"[{c['plan']}, p.{c['page']}]\n{c['text']}" for c in chunks)


def answer_question(question: str, plan: str | None = None, top_k: int = 4) -> dict:
    chunks = retrieve(question, top_k=top_k, plan=plan)
    prompt = (
        f"Context:\n{build_context(chunks)}\n\nQuestion: {question}\n\n"
        "Answer using only the context:"
    )
    return {
        "answer": ask_llm(prompt),
        "sources": [
            {"plan": c["plan"], "page": c["page"], "similarity": round(c["similarity"], 3)}
            for c in chunks
        ],
    }
