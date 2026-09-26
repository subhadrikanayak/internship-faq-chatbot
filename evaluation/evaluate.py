"""
evaluation/evaluate.py

Loads the ACTUAL trained model + processed combined dataset, recreates the
exact same stratified train/test split used during training (same
random_state), and computes real evaluation metrics on the held-out test
set:

    - Accuracy
    - Precision / Recall / F1 (macro & weighted)
    - Full classification report (all 47 intents: 27 customer-service +
      20 internship)
    - Confusion matrix (saved as an image; too large to annotate at
      47x47, so it is saved unannotated - see classification_report.txt
      for exact per-class numbers)

Run from the project root with:
    python -m evaluation.evaluate

Nothing in this file is hardcoded - every number is produced by actually
running the trained model against the test split.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from backend.config import settings
from backend.utils import load_classifier, load_intent_domain_map, load_label_encoder, load_vectorizer


def main():
    print("Loading processed combined dataset and trained model artifacts...")
    df = pd.read_csv(settings.PROCESSED_DATA_PATH)

    vectorizer = load_vectorizer()
    classifier = load_classifier()
    label_encoder = load_label_encoder()
    domain_map = load_intent_domain_map()

    X = df["clean_text"].values
    y = label_encoder.transform(df["intent"].values)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y,
    )

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n================ EVALUATION RESULTS (all intents) ================")
    print(f"Test set size: {len(y_test)}")
    print(f"Accuracy:             {acc:.4f}")
    print(f"Precision (macro):    {precision_macro:.4f}")
    print(f"Recall (macro):       {recall_macro:.4f}")
    print(f"F1-score (macro):     {f1_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted):    {recall_weighted:.4f}")
    print(f"F1-score (weighted):  {f1_weighted:.4f}")
    print("====================================================================\n")

    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    print("Full classification report (all intents):\n")
    print(report)

    # --- Internship-only subset metrics (the intents this app actually
    # answers from the FAQ) - useful because the overall accuracy above is
    # dominated by the much larger Kaggle customer-service classes. ---
    internship_intents = {i for i, d in domain_map.items() if d == "internship"}
    mask = pd.Series(label_encoder.inverse_transform(y_test)).isin(internship_intents)
    if mask.sum() > 0:
        y_test_i = y_test[mask.values]
        y_pred_i = y_pred[mask.values]
        acc_i = accuracy_score(y_test_i, y_pred_i)
        print(f"\nInternship-only subset - test examples: {mask.sum()}, "
              f"accuracy: {acc_i:.4f}")
        print("(Small per-class counts for internship intents are expected - "
              "see README Limitations.)\n")

    settings.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = settings.EVALUATION_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("EVALUATION RESULTS (all 47 intents: customer-service + internship)\n")
        f.write("=====================================================================\n")
        f.write(f"Test set size: {len(y_test)}\n")
        f.write(f"Accuracy:             {acc:.4f}\n")
        f.write(f"Precision (macro):    {precision_macro:.4f}\n")
        f.write(f"Recall (macro):       {recall_macro:.4f}\n")
        f.write(f"F1-score (macro):     {f1_macro:.4f}\n")
        f.write(f"Precision (weighted): {precision_weighted:.4f}\n")
        f.write(f"Recall (weighted):    {recall_weighted:.4f}\n")
        f.write(f"F1-score (weighted):  {f1_weighted:.4f}\n\n")
        f.write(report)
        if mask.sum() > 0:
            f.write(f"\nInternship-only subset - test examples: {mask.sum()}, "
                    f"accuracy: {acc_i:.4f}\n")
    print(f"Saved text report to: {report_path}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(22, 20))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("Confusion Matrix - Combined Intent Classifier (Test Set)")
    plt.xlabel("Predicted Intent")
    plt.ylabel("True Intent")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    cm_path = settings.EVALUATION_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix image to: {cm_path}")


if __name__ == "__main__":
    main()
