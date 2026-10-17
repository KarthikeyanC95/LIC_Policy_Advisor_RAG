import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"            # public LIC brochure PDFs go here
CHROMA_DIR = DATA_DIR / "chroma"      # vector index (built by scripts/ingest.py)
POLICIES_CSV = DATA_DIR / "policies.csv"
RATES_CSV = DATA_DIR / "premium_rates.csv"

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
EMBED_MODEL = "all-MiniLM-L6-v2"

DISCLAIMER = (
    "General information only, not financial advice. Confirm plan terms and "
    "premium with LIC or a licensed agent before buying."
)
