"""
backend/config.py

Central configuration for the Internship FAQ & AI Assistant backend.
Loads secrets/settings from environment variables (.env file).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    # --- Gemini API ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    # --- Paths: data ---
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = DATA_DIR / "raw" / (
        "20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv"
    )
    INTERNSHIP_FAQ_PATH: Path = DATA_DIR / "internship_faq" / "internship_faq.json"
    PROCESSED_DATA_PATH: Path = DATA_DIR / "processed" / "combined_training_dataset.csv"

    # --- Paths: models ---
    MODELS_DIR: Path = BASE_DIR / "models"
    VECTORIZER_PATH: Path = MODELS_DIR / "vectorizer.pkl"
    CLASSIFIER_PATH: Path = MODELS_DIR / "intent_classifier.pkl"
    LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.pkl"
    INTENT_DOMAIN_MAP_PATH: Path = MODELS_DIR / "intent_domain_map.json"

    EVALUATION_DIR: Path = BASE_DIR / "evaluation"

    # --- Model / training ---
    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))

    # --- Confidence threshold ---
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

    # --- CORS ---
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


settings = Settings()