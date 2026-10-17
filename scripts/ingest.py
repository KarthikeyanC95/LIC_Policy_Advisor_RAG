"""Build the vector index from data/raw/*.pdf and create a blank policies.csv template.

Run:  python -m scripts.ingest
"""
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from app import config, rag


def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def main():
    pdfs = sorted(Path(config.RAW_DIR).rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {config.RAW_DIR}. Add public LIC brochures first.")

    records = []
    for pdf in pdfs:
        for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
            text = (page.extract_text() or "").strip()
            for i, c in enumerate(chunk_text(text) if text else []):
                records.append({"id": f"{pdf.stem}_p{page_no}_c{i}", "text": c, "plan": pdf.stem, "page": page_no})
    print(f"{len(pdfs)} PDFs -> {len(records)} chunks")

    col = rag.collection()
    existing = col.get()["ids"]
    if existing:
        col.delete(ids=existing)  # rebuild cleanly
    for i in range(0, len(records), 500):
        b = records[i:i + 500]
        col.add(
            ids=[r["id"] for r in b],
            documents=[r["text"] for r in b],
            metadatas=[{"plan": r["plan"], "page": r["page"]} for r in b],
            embeddings=rag.embed_texts([r["text"] for r in b]),
        )
    print("Indexed:", col.count())

    if not config.POLICIES_CSV.exists():
        t = pd.DataFrame({"plan": sorted({p.stem for p in pdfs})})
        for c in ["goal", "min_age", "max_age", "min_term", "max_term", "min_sum_assured", "max_sum_assured"]:
            t[c] = ""
        t.to_csv(config.POLICIES_CSV, index=False)
        print(f"Created {config.POLICIES_CSV}. Fill it from the brochures.")


if __name__ == "__main__":
    main()
