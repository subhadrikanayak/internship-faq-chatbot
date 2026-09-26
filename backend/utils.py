"""
backend/utils.py

Shared helper utilities:
    - Internship FAQ knowledge base loading/lookup
    - Trained model loading (vectorizer, classifier, label encoder)
    - Intent -> domain map loading (generated at training time from the
      ACTUAL combined training data, not hardcoded here)
"""

import json

import joblib

from backend.config import settings

# ---------------------------------------------------------------------------
# INTERNSHIP FAQ KNOWLEDGE BASE
# ---------------------------------------------------------------------------

_faq_cache = None


def load_internship_faq() -> dict:
    """
    Load data/internship_faq/internship_faq.json and return it indexed by
    intent name for O(1) lookup:

        {
            "eligibility": {"questions": [...], "answer": "..."},
            ...
        }
    """
    global _faq_cache
    if _faq_cache is None:
        with open(settings.INTERNSHIP_FAQ_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        _faq_cache = {entry["intent"]: entry for entry in raw["faqs"]}
    return _faq_cache


def get_faq_answer(intent: str) -> str:
    """Return the safe, generic FAQ answer/context for a given internship intent."""
    faq = load_internship_faq()
    entry = faq.get(intent)
    if entry is None:
        return "No specific FAQ context is available for this intent."
    return entry["answer"]


# ---------------------------------------------------------------------------
# MODEL LOADING HELPERS
# ---------------------------------------------------------------------------

def load_vectorizer():
    """Load the trained TF-IDF vectorizer from disk."""
    return joblib.load(settings.VECTORIZER_PATH)


def load_classifier():
    """Load the trained Logistic Regression classifier from disk."""
    return joblib.load(settings.CLASSIFIER_PATH)


def load_label_encoder():
    """Load the label encoder (maps class indices <-> intent strings)."""
    return joblib.load(settings.LABEL_ENCODER_PATH)


def load_intent_domain_map() -> dict:
    """
    Load the intent -> domain mapping produced by backend/train.py from
    the actual combined training dataset (e.g. "eligibility" -> "internship",
    "track_order" -> "customer_service"). This is NOT hand-written here -
    it reflects exactly what domain each intent had in the training data.
    """
    with open(settings.INTENT_DOMAIN_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
