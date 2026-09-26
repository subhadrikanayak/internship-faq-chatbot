"""
backend/gemini_service.py

Thin wrapper around the Google Gemini API using the current official
`google-genai` Python SDK.

Responsibility of this module ONLY:
    Given the user's question, the detected intent, confidence, domain,
    and (when available) internship FAQ context, generate the final
    natural-language reply.

Gemini does NOT decide the intent or domain - that is the job of the
TF-IDF + Logistic Regression classifier plus the intent->domain map in
backend/chatbot.py. Gemini turns that decision into a natural sentence,
and:
    - for internship questions, must not invent policies/deadlines/fees/
      stipend amounts/mentor names/contact details beyond the FAQ context
    - for general questions, answers normally and helpfully
    - for current/real-time data (weather, stock prices, live scores),
      must not fabricate current information
"""

import time

from google import genai
from google.genai import types

from backend.config import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Please add it to your .env file."
            )
        _client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            # Short, explicit timeout (ms): the assistant must always
            # respond promptly, even if the Gemini call fails or is slow.
            http_options=types.HttpOptions(timeout=35_000),
        )
    return _client


SYSTEM_PROMPT = """You are a helpful AI assistant that serves as an Internship FAQ \
Assistant and General AI Assistant.

For internship-related questions, use the provided internship FAQ context and \
detected intent to answer accurately.

Do not invent internship policies, deadlines, fees, stipend amounts, mentor \
information, contact information, or other organization-specific facts.

If the required internship information is not available in the provided FAQ \
context, clearly say that the information is not available and recommend checking \
the official internship documentation or contacting the internship coordinator.

For general questions that are outside the internship FAQ domain, answer normally \
and helpfully, using your own general knowledge.

Do not force general questions into internship categories.

Do not say that you cannot understand a question simply because it is outside the \
training dataset - always give a useful, substantive response.

For current or real-time information such as weather, stock prices, live sports \
scores, or breaking news, do not fabricate current information when no live data \
source is connected. Clearly explain that current/live data requires a connected \
real-time source instead.

Keep answers clear, natural, and reasonably concise (1-5 sentences unless the \
question genuinely needs more detail).
"""


def _fallback_text(domain: str, intent: str) -> str:
    """Used only if the Gemini API call itself fails (network/quota/key issue) -
    the assistant must still respond, never crash or go silent."""
    if domain == "internship":
        return (
            "I believe this is related to the internship program, but I'm having "
            "trouble reaching the AI response service right now. In general, for "
            "questions like this, please check the official internship notice or "
            "contact the internship coordinator for accurate details."
        )
    return (
        "I'm having trouble reaching the AI response service right now. Could you "
        "try again in a moment? If this is an internship-related question, checking "
        "the official internship notice or contacting the coordinator is a good "
        "next step in the meantime."
    )


def generate_response(
    user_message: str,
    intent: str,
    confidence: float,
    domain: str,
    faq_context: str | None,
) -> str:
    """
    Call the Gemini API to produce the final natural-language response.

    Args:
        user_message: The raw text the user typed.
        intent: The predicted intent label, or "general" when the question
            falls outside the internship FAQ domain (low confidence, or a
            confidently-predicted intent from the customer-service domain
            that this internship assistant does not handle).
        confidence: The classifier's confidence score (0-1) for the predicted intent.
        domain: "internship" or "general".
        faq_context: The internship FAQ answer/context to ground the reply on,
            or None when domain == "general".

    Returns:
        The generated natural-language response string.
    """
    client = _get_client()

    if domain == "internship" and faq_context:
        prompt = f"""User question: "{user_message}"

Detected intent: {intent}
Confidence: {confidence:.2f}
Domain: internship

Internship FAQ context (use ONLY this information for specific facts):
{faq_context}

Using the FAQ context above, write a natural, helpful reply to the user's question. \
If the FAQ context says specific information (like an exact deadline, fee, or name) \
is not available, pass that along clearly rather than inventing a number or name."""
    else:
        prompt = f"""User question: "{user_message}"

This question does not confidently match a known internship FAQ topic (domain: \
general, confidence for the closest matched category was {confidence:.2f}, below \
the configured threshold, or it matched a category outside the internship FAQ \
domain). Answer the user's question normally and helpfully using your own general \
knowledge. Do not force this into an internship category. If the question asks for \
current/live information (e.g. today's weather, current stock price), explain that \
you don't have a connected real-time data source instead of guessing a value."""

    RETRYABLE_MARKERS = (
        "429",
        "503",
        "resource_exhausted",
        "unavailable",
        "timed out",
        "timeout",
        "connectionerror",
        "connection reset",
    )
    max_attempts = 3
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.4,
                    # NOTE: on Gemini 3.x models, max_output_tokens is a
                    # COMBINED budget for internal "thinking" tokens plus
                    # the visible answer - a low value here can get eaten
                    # almost entirely by thinking, cutting the visible
                    # reply off mid-sentence. Raised well above what a
                    # short chatbot answer needs, and thinking_level is
                    # capped to "low" so most of the budget goes to the
                    # actual answer instead of invisible reasoning.
                    max_output_tokens=2048,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise ValueError("Empty response from Gemini")
            return text
        except Exception as exc:  # noqa: BLE001 - always respond, never crash
            last_exc = exc
            is_retryable = any(
                marker in str(exc).lower() for marker in RETRYABLE_MARKERS
            )
            if is_retryable and attempt < max_attempts:
                wait_seconds = 2 * attempt  # 2s, then 4s
                print(
                    f"[gemini_service] Transient error on attempt {attempt}/"
                    f"{max_attempts} ({exc}). Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
                continue
            print(f"[gemini_service] Gemini API call failed: {exc}")
            return _fallback_text(domain, intent)

    # Should be unreachable, but keep the function total just in case.
    print(f"[gemini_service] Gemini API call failed after retries: {last_exc}")
    return _fallback_text(domain, intent)