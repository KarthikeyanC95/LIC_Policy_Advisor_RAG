import pandas as pd

from app.advisor import clean_policies, eligible_plans, estimate_premium

# Synthetic test data. These are NOT real LIC plan terms.
POLICIES = pd.DataFrame([
    {"plan": "plan_a", "goal": "protection", "min_age": 18, "max_age": 65, "min_term": 10, "max_term": 40,
     "min_sum_assured": 100000, "max_sum_assured": None},
    {"plan": "plan_b", "goal": "savings", "min_age": 18, "max_age": 55, "min_term": 12, "max_term": 25,
     "min_sum_assured": 200000, "max_sum_assured": 5000000},
    {"plan": "plan_incomplete", "goal": "", "min_age": None, "max_age": None, "min_term": None, "max_term": None,
     "min_sum_assured": None, "max_sum_assured": None},
])


def P(**kw):
    base = {"age": 30, "goal": "protection", "term": 20, "sum_assured": 1000000}
    return {**base, **kw}


def test_matching_plan_is_returned():
    assert list(eligible_plans(P(), POLICIES).plan) == ["plan_a"]


def test_goal_is_case_insensitive():
    assert list(eligible_plans(P(goal="  Protection "), POLICIES).plan) == ["plan_a"]


def test_age_boundaries_are_inclusive():
    assert not eligible_plans(P(age=18), POLICIES).empty
    assert not eligible_plans(P(age=65), POLICIES).empty
    assert eligible_plans(P(age=17), POLICIES).empty
    assert eligible_plans(P(age=66), POLICIES).empty


def test_blank_max_sum_assured_means_no_limit():
    assert not eligible_plans(P(sum_assured=900_000_000), POLICIES).empty


def test_max_sum_assured_is_enforced():
    assert eligible_plans(P(goal="savings", sum_assured=5_000_001), POLICIES).empty
    assert not eligible_plans(P(goal="savings", sum_assured=5_000_000), POLICIES).empty


def test_incomplete_rows_are_skipped_and_reported():
    usable, skipped = clean_policies(POLICIES)
    assert "plan_incomplete" in skipped
    assert "plan_incomplete" not in list(usable.plan)


def test_premium_lookup_and_missing():
    rates = pd.DataFrame([{"plan": "plan_a", "age": 30, "term": 20, "annual_rate_per_1000": 2.5}])
    assert estimate_premium("plan_a", 30, 20, 1_000_000, rates) == 2500.0
    assert estimate_premium("plan_a", 31, 20, 1_000_000, rates) is None
