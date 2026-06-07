"""
Credit risk scoring module.

Loads the trained XGBoost model and exposes a single clean entry point,
score_applicant(), that the API and agent layers import. Returns a risk
probability, a decision, and SHAP-based per-applicant explanations suitable
for adverse-action-style reasoning.
"""

from pathlib import Path
import joblib
import pandas as pd
import shap

# --- Load artifacts once at import time (production serving pattern) ---
# Expensive objects (model, explainer) are built once and reused per call,
# not rebuilt on every request.
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_model = joblib.load(_MODELS_DIR / "credit_risk_model.pkl")
_feature_columns = joblib.load(_MODELS_DIR / "feature_columns.pkl")
_explainer = shap.TreeExplainer(_model)

DEFAULT_THRESHOLD = 0.5


def encode_applicant(applicant: dict) -> pd.DataFrame:
    """Turn a raw applicant dict into a model-ready, column-aligned row.

    The reindex against the saved feature columns is the defense against
    train-serving skew: missing categories become 0, extras are dropped,
    and column order matches training exactly.
    """
    raw = pd.DataFrame([applicant])
    encoded = pd.get_dummies(raw)
    aligned = encoded.reindex(columns=_feature_columns, fill_value=0)
    return aligned


def score_applicant(applicant: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Score a loan applicant.

    Args:
        applicant: raw applicant fields (see sample in README/tests).
        threshold: probability cutoff for DECLINE. Exposed because it is a
            business lever, not a fixed property of the model.

    Returns:
        dict with probability, risk tier, decision, signed SHAP factors,
        and which explanation method was used.
    """
    X = encode_applicant(applicant)
    prob_default = float(_model.predict_proba(X)[0, 1])

    decision = "DECLINE" if prob_default >= threshold else "APPROVE"
    if prob_default < 0.3:
        tier = "Low"
    elif prob_default < 0.6:
        tier = "Medium"
    else:
        tier = "High"

    try:
        shap_values = _explainer.shap_values(X)[0]
        contribs = sorted(
            zip(_feature_columns, shap_values),
            key=lambda x: abs(x[1]), reverse=True
        )
        factors_increasing_risk = [f for f, v in contribs if v > 0][:3]
        factors_decreasing_risk = [f for f, v in contribs if v < 0][:3]
        method = "shap"
    except Exception as e:
        importances = dict(zip(_feature_columns, _model.feature_importances_))
        active = X.iloc[0]
        active_feats = [f for f in _feature_columns if active[f] != 0]
        factors_increasing_risk = sorted(
            active_feats, key=lambda f: importances[f], reverse=True
        )[:3]
        factors_decreasing_risk = []
        method = f"fallback ({type(e).__name__})"

    return {
        "probability_of_default": round(prob_default, 3),
        "risk_tier": tier,
        "decision": decision,
        "threshold_used": threshold,
        "factors_increasing_risk": factors_increasing_risk,
        "factors_decreasing_risk": factors_decreasing_risk,
        "explanation_method": method,
    }


# Sample applicant for quick manual testing
SAMPLE_APPLICANT = {
    "checking_account": "A11", "duration_months": 24, "credit_history": "A32",
    "purpose": "A43", "credit_amount": 5000, "savings_account": "A61",
    "employment_since": "A73", "installment_rate": 4, "personal_status_sex": "A93",
    "other_debtors": "A101", "residence_since": 2, "property": "A121", "age": 35,
    "other_installment_plans": "A143", "housing": "A152", "existing_credits": 1,
    "job": "A173", "num_dependents": 1, "telephone": "A192", "foreign_worker": "A201",
}


if __name__ == "__main__":
    import json
    print(json.dumps(score_applicant(SAMPLE_APPLICANT), indent=2))
