# LIC Policy Advisor RAG

A FastAPI service that answers questions about LIC policy brochures, filters plans a customer is eligible for, and estimates premiums, with page-level citations.

**Design rule:** the LLM only explains benefits from retrieved brochure text. Eligibility rules and premium maths run in plain Python from tables transcribed from the brochures. The model never calculates or guesses money.

> General information only, not financial advice. Use public brochures only. Confirm plan terms and premiums with LIC or a licensed agent.

## Architecture

```text
PDF brochures -> chunk -> embeddings (MiniLM) -> ChromaDB
Question -> retrieve top-k (optionally within one plan) -> Groq LLM -> answer + citations
Profile  -> eligibility filter (pandas) -> per-plan benefits (RAG) -> premium lookup (pandas) -> budget check
FastAPI -> Docker -> GitHub Actions -> GitHub Container Registry
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /health | Status and number of indexed chunks |
| GET | /plans | Plans with complete vs incomplete eligibility data |
| POST | /ask | Question answering with citations |
| POST | /recommend | Eligible plans, benefits and premium estimate for a profile |

Interactive docs: http://localhost:8000/docs

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...                             # Windows: set GROQ_API_KEY=gsk_...

# 1. put public brochure PDFs in data/raw/
python -m scripts.ingest                                # builds the index and a blank data/policies.csv
# 2. fill data/policies.csv (and optionally data/premium_rates.csv) from the brochures
uvicorn app.main:app --reload
```

`data/policies.csv` columns: `plan, goal, min_age, max_age, min_term, max_term, min_sum_assured, max_sum_assured` (leave max blank for no limit).
`data/premium_rates.csv` columns: `plan, age, term, annual_rate_per_1000` (rate per Rs 1,000 sum assured).

## Run with Docker

```bash
cp .env.example .env            # add your Groq key
docker compose build
docker compose run --rm api python -m scripts.ingest
docker compose up
```

The `data/` folder is mounted into the container, so PDFs, the index and the CSVs stay out of the image and out of Git.

## Try it

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is paid on death during the policy term?"}'

curl -X POST localhost:8000/recommend -H "Content-Type: application/json" \
  -d '{"age": 30, "goal": "protection", "term": 20, "sum_assured": 5000000, "annual_budget": 30000}'
```

## Tests and CI

```bash
pip install pandas pytest && pytest -q
```

Tests cover eligibility boundaries, blank limits, incomplete rows and premium lookup using synthetic data. GitHub Actions runs them on every push, builds the Docker image, and on `main` publishes it to GitHub Container Registry as `ghcr.io/<your-github-username>/<repo-name>` (lowercased automatically).

## Evaluation

Document your results here before publishing:

- Test set: 30 questions (22 answerable, 8 out of scope)
- Retrieval hit rate at k=4: [X%]
- Citation accuracy: [X%]
- Correct refusals on out-of-scope questions: [X%]
- Wrong or invented numbers: [0]
- Chunk size / top_k comparison table
- Failure analysis: 3 wrong answers and their causes

## Limitations

- Plan ranking is simple (alphabetical among eligible plans). Add scoring by goal fit and budget as a next step.
- PDF tables can extract badly, so premiums come from a transcribed table, not the LLM.
- No authentication or rate limiting yet.
