"""Eligibility filter and premium lookup. Plain pandas, no LLM: money and rules are never left to a model."""
import pandas as pd

from . import config

NUM_COLS = ["min_age", "max_age", "min_term", "max_term", "min_sum_assured", "max_sum_assured"]


def load_policies() -> pd.DataFrame:
    if not config.POLICIES_CSV.exists():
        return pd.DataFrame(columns=["plan", "goal", *NUM_COLS])
    return pd.read_csv(config.POLICIES_CSV)


def load_rates() -> pd.DataFrame | None:
    return pd.read_csv(config.RATES_CSV) if config.RATES_CSV.exists() else None


def clean_policies(df: pd.DataFrame):
    """Return (usable rows, names of plans skipped because the table is incomplete)."""
    df = df.copy()
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["max_sum_assured"] = df["max_sum_assured"].fillna(float("inf"))  # blank = no upper limit
    df["goal"] = df["goal"].fillna("").astype(str).str.strip().str.lower()
    required = [c for c in NUM_COLS if c != "max_sum_assured"]
    bad = (df["goal"] == "") | df[required].isna().any(axis=1)
    return df[~bad], list(df.loc[bad, "plan"])


def eligible_plans(profile: dict, policies: pd.DataFrame | None = None) -> pd.DataFrame:
    df, _ = clean_policies(load_policies() if policies is None else policies)
    ok = (
        (df.min_age <= profile["age"]) & (df.max_age >= profile["age"])
        & (df.min_term <= profile["term"]) & (df.max_term >= profile["term"])
        & (df.min_sum_assured <= profile["sum_assured"])
        & (df.max_sum_assured >= profile["sum_assured"])
        & (df.goal == str(profile["goal"]).strip().lower())
    )
    return df[ok].sort_values("plan")


def estimate_premium(plan, age, term, sum_assured, rates: pd.DataFrame | None = None):
    """Annual premium = rate per Rs 1,000 x sum assured / 1,000. None if the plan/age/term is not in the table."""
    rates = load_rates() if rates is None else rates
    if rates is None:
        return None
    row = rates[(rates.plan == plan) & (rates.age == age) & (rates.term == term)]
    if row.empty:
        return None
    return float(row.iloc[0].annual_rate_per_1000) * sum_assured / 1000


def recommend(profile: dict, top_n: int = 3) -> dict:
    from . import rag  # lazy: keeps this module importable without the ML stack

    _, skipped = clean_policies(load_policies())
    elig = eligible_plans(profile)
    plans = []
    for plan in list(elig.plan.head(top_n)):
        chunks = rag.retrieve(
            "benefits: death benefit, maturity benefit, bonuses, riders, exclusions, surrender, loan",
            top_k=5, plan=plan,
        )
        summary = rag.ask_llm(
            f"Context:\n{rag.build_context(chunks)}\n\nCustomer goal: {profile['goal']}.\n"
            "Summarise in 4 bullet points: main benefits, what is paid on death, what is paid at maturity, "
            "and important conditions or exclusions. Cite pages."
        )
        prem = estimate_premium(plan, profile["age"], profile["term"], profile["sum_assured"])
        budget = profile.get("annual_budget")
        plans.append({
            "plan": plan,
            "benefits_summary": summary,
            "estimated_annual_premium": prem,
            "premium_note": (
                "Estimate before GST, rebates and payment-mode changes."
                if prem is not None
                else "Not in my premium table. Use LIC's official premium calculator."
            ),
            "within_budget": None if (prem is None or not budget) else prem <= budget,
        })
    return {"plans": plans, "skipped_incomplete_rows": skipped, "disclaimer": config.DISCLAIMER}
