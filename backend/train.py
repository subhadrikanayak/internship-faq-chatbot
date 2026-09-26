"""
backend/train.py

End-to-end training script for the unified intent classifier.

WHY A UNIFIED CLASSIFIER (design decision, explained for the mentor demo):
Two options were considered:
    (a) Train two separate classifiers (one for customer-service intents
        from the Kaggle dataset, one for internship intents).
    (b) Train ONE classifier on a combined dataset that keeps a clear
        `domain` and `source` column per row, so the model learns to
        distinguish internship vocabulary ("internship", "stipend",
        "mentor", "certificate"...) from customer-service vocabulary
        ("order", "refund", "invoice"...) as part of the same decision
        boundary.
Option (b) was chosen because it is simpler for a beginner-level project
to build, train, save and demonstrate (ONE vectorizer + ONE classifier +
ONE training script), while the `domain` column - saved separately as
models/intent_domain_map.json - still lets the chatbot logic (backend/
chatbot.py) know whether a predicted intent belongs to the internship
FAQ domain or not, without needing a second model.

Steps:
    1. Load the Kaggle customer-service dataset (raw CSV)
    2. Load and expand the internship FAQ dataset (JSON -> rows)
    3. Combine both into one dataframe with `domain` and `source` columns
    4. Validate the combined dataset (missing values, duplicates, class counts)
    5. Preprocess all text
    6. Train/test split (stratified by intent, random_state=42)
    7. Fit TF-IDF vectorizer
    8. Train Logistic Regression classifier
    9. Evaluate on the test set
    10. Save vectorizer, classifier, label encoder, and intent->domain map

Run from the project root with:
    python -m backend.train
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from backend.config import settings
from backend.preprocess import preprocess_text


def load_kaggle_dataset() -> pd.DataFrame:
    print(f"Loading Kaggle customer-service dataset from: {settings.RAW_DATA_PATH}")
    df = pd.read_csv(settings.RAW_DATA_PATH)
    print(f"Kaggle dataset loaded. Shape: {df.shape}")

    df = df.rename(columns={"utterance": "text"})
    df = df[["text", "intent"]].copy()
    df["domain"] = "customer_service"
    df["source"] = "bitext"
    return df


def load_internship_dataset() -> pd.DataFrame:
    print(f"Loading internship FAQ dataset from: {settings.INTERNSHIP_FAQ_PATH}")
    with open(settings.INTERNSHIP_FAQ_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for entry in raw["faqs"]:
        intent = entry["intent"]
        for question in entry["questions"]:
            rows.append(
                {
                    "text": question,
                    "intent": intent,
                    "domain": "internship",
                    "source": "custom_faq",
                }
            )
    df = pd.DataFrame(rows)
    print(f"Internship FAQ dataset expanded to {len(df)} example questions "
          f"across {df['intent'].nunique()} internship intents.")
    return df


def build_combined_dataset() -> pd.DataFrame:
    kaggle_df = load_kaggle_dataset()
    internship_df = load_internship_dataset()

    combined = pd.concat([kaggle_df, internship_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text", "intent"]).reset_index(drop=True)

    print(f"\nCombined dataset shape (Kaggle + internship): {combined.shape}")
    return combined


def validate_dataset(df: pd.DataFrame) -> None:
    print("\n--- DATASET VALIDATION ---")

    print("\nMissing values per column:")
    print(df.isnull().sum())

    dup_rows = df.duplicated(subset=["text"]).sum()
    print(f"\nDuplicate text entries: {dup_rows}")

    n_intents = df["intent"].nunique()
    n_domains = df["domain"].nunique()
    print(f"\nNumber of unique intents (both domains combined): {n_intents}")
    print(f"Number of domains: {n_domains} -> {sorted(df['domain'].unique())}")

    print("\nExamples per domain:")
    print(df["domain"].value_counts())

    print("\nExamples per intent (sorted, smallest first - internship intents "
          "will naturally have far fewer examples than Kaggle intents):")
    print(df["intent"].value_counts().sort_values())

    print("\n--- END VALIDATION ---\n")


def build_processed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("Applying preprocessing pipeline to all text "
          "(this may take a little while for 21k+ rows)...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    before = len(df)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows that became empty after preprocessing.")

    settings.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.PROCESSED_DATA_PATH, index=False)
    print(f"Processed combined dataset saved to: {settings.PROCESSED_DATA_PATH}")

    return df


def save_intent_domain_map(df: pd.DataFrame) -> None:
    """
    Build the intent -> domain lookup directly from the ACTUAL combined
    training data (not hand-written), so it always reflects what the
    model was actually trained on. Saved as JSON for the chatbot logic
    to load at inference time.
    """
    mapping = (
        df.groupby("intent")["domain"]
        .agg(lambda s: s.mode().iloc[0])  # each intent should have exactly one domain
        .to_dict()
    )
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(settings.INTENT_DOMAIN_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    print(f"Saved intent->domain map ({len(mapping)} intents) to: "
          f"{settings.INTENT_DOMAIN_MAP_PATH}")


def train():
    combined_df = build_combined_dataset()
    validate_dataset(combined_df)
    combined_df = build_processed_dataset(combined_df)

    save_intent_domain_map(combined_df)

    X = combined_df["clean_text"].values
    y_raw = combined_df["intent"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print(f"\nTotal number of intent classes (both domains): {len(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y,
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,   # kept low (1) so the small internship intents aren't pruned away
        max_df=0.9,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    classifier = LogisticRegression(
        max_iter=1000,
        C=10.0,
        random_state=settings.RANDOM_STATE,
        class_weight="balanced",
    )
    print("\nTraining Logistic Regression classifier on the combined dataset...")
    classifier.fit(X_train_tfidf, y_train)

    y_pred = classifier.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy (all {len(label_encoder.classes_)} intents combined): {acc:.4f}")
    print("\nClassification report (test set):")
    print(
        classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
        )
    )

    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, settings.VECTORIZER_PATH)
    joblib.dump(classifier, settings.CLASSIFIER_PATH)
    joblib.dump(label_encoder, settings.LABEL_ENCODER_PATH)

    print(f"\nSaved vectorizer to:    {settings.VECTORIZER_PATH}")
    print(f"Saved classifier to:    {settings.CLASSIFIER_PATH}")
    print(f"Saved label encoder to: {settings.LABEL_ENCODER_PATH}")
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
