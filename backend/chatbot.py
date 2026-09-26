"""
backend/chatbot.py

Core chatbot logic. Ties together:
    1. Preprocessing
    2. TF-IDF vectorization
    3. Logistic Regression intent prediction + confidence (predict_proba)
    4. Confidence-threshold decision
    5. Intent -> domain lookup (from the ACTUAL training data, see
       models/intent_domain_map.json, produced by backend/train.py)
    6. Internship FAQ context retrieval (only for confident internship intents)
    7. Gemini response generation

DECISION LOGIC (matches the project spec exactly):

    predict intent + confidence with the unified classifier
    if confidence >= threshold AND intent's domain == "internship":
        -> answer using the internship FAQ context for that intent
    else:
        -> intent reported to the user as "general"
           (this covers: low confidence, OR a confident match against one
           of the Kaggle customer-service intents, which this internship
           assistant does not serve)
        -> Gemini answers the raw question directly, no FAQ context

This module is kept separate from main.py (the FastAPI routing layer) so
the logic is easy to test and easy to explain to a mentor independently
of the web framework.
"""

from dataclasses import dataclass

import numpy as np

from backend.config import settings
from backend.gemini_service import generate_response
from backend.preprocess import preprocess_text
from backend.utils import (
    get_faq_answer,
    load_classifier,
    load_intent_domain_map,
    load_label_encoder,
    load_vectorizer,
)


@dataclass
class ChatResult:
    response: str
    intent: str
    confidence: float
    domain: str


class ChatbotEngine:
    """
    Loads the trained vectorizer, classifier, label encoder, and the
    intent->domain map ONCE at startup, and reuses them for every
    incoming request instead of re-reading files from disk each time.
    """

    def __init__(self):
        print("[chatbot] Loading vectorizer, classifier, label encoder, "
              "and intent->domain map...")
        self.vectorizer = load_vectorizer()
        self.classifier = load_classifier()
        self.label_encoder = load_label_encoder()
        self.intent_domain_map = load_intent_domain_map()
        print("[chatbot] Models loaded successfully. "
              f"{len(self.intent_domain_map)} intents known "
              f"({sum(1 for d in self.intent_domain_map.values() if d == 'internship')} "
              "internship, "
              f"{sum(1 for d in self.intent_domain_map.values() if d == 'customer_service')} "
              "customer-service).")

    def predict_intent(self, clean_text: str):
        """Returns (predicted_intent: str, confidence: float)."""
        X = self.vectorizer.transform([clean_text])
        probs = self.classifier.predict_proba(X)[0]

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        predicted_intent = self.label_encoder.inverse_transform([best_idx])[0]

        return predicted_intent, confidence

    def handle_message(self, user_message: str, threshold: float = None) -> ChatResult:
        if threshold is None:
            threshold = settings.CONFIDENCE_THRESHOLD

        user_message = (user_message or "").strip()
        if not user_message:
            return ChatResult(
                response="Could you type your question? I didn't receive any text.",
                intent="general",
                confidence=0.0,
                domain="general",
            )

        clean_text = preprocess_text(user_message)

        if not clean_text:
            # Message was only punctuation/emojis/etc. after cleaning.
            raw_intent, confidence = "general", 0.0
        else:
            raw_intent, confidence = self.predict_intent(clean_text)

        predicted_domain = self.intent_domain_map.get(raw_intent, "customer_service")

        is_confident = confidence >= threshold
        is_internship_match = is_confident and predicted_domain == "internship"

        if is_internship_match:
            final_intent = raw_intent
            final_domain = "internship"
            faq_context = get_faq_answer(raw_intent)
        else:
            # Either low confidence, or a confident customer-service match -
            # this internship assistant treats both as a general question.
            final_intent = "general"
            final_domain = "general"
            faq_context = None

        final_response = generate_response(
            user_message=user_message,
            intent=final_intent,
            confidence=confidence,
            domain=final_domain,
            faq_context=faq_context,
        )

        return ChatResult(
            response=final_response,
            intent=final_intent,
            confidence=round(confidence, 4),
            domain=final_domain,
        )


_engine: ChatbotEngine | None = None


def get_engine() -> ChatbotEngine:
    global _engine
    if _engine is None:
        _engine = ChatbotEngine()
    return _engine
