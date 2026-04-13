"""Evaluation metric implementations."""


def score_exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 when normalized strings match, else 0.0."""
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
