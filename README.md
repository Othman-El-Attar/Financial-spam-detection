# FinSight SMS Security Lab — optional sender

Deploy the **contents** of this folder at the Vercel repository root. Python endpoint: `/api/predict`. The sender ID is optional. Known sender IDs are exact-match aliases in `api/predict.py`; this list is illustrative and must be verified against official sender records before production use. A numeric phone sender (7–15 digits, optional + and formatting) is flagged `spam` by the requested strict rule. Otherwise the existing ML model predicts from SMS text and optional sender ID. `sender_status` and `decision_source` explain the result. A rule-only result has `probabilities: null` because it is not a calibrated ML score.

The existing model was trained on synthetic data; `safe` does not establish authenticity. Keep the deployed scikit-learn version compatible with the exported joblib model. This package includes the previously required compatibility shim.
