# Internship FAQ & General AI Assistant (NLP + FastAPI + Gemini)

## 1. Project Objective

An assistant that:
1. Answers **internship FAQ questions** (eligibility, application process, duration, stipend, certificate, etc.) using a trained NLP intent classifier + an internship FAQ knowledge base.
2. Answers **general questions** ("What is Python?", "Tell me a joke") using Gemini's own knowledge.
3. Never just says "I don't understand" — every message gets a real, useful response.
4. Still demonstrates a complete, from-scratch ML pipeline (preprocessing → TF-IDF → Logistic Regression → evaluation) for the internship/mentor demo.

## 2. Why the Kaggle Dataset Is Used

The Kaggle Bitext dataset (`20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv`) is a **customer-service** dataset — 21,534 rows, columns `flags`/`utterance`/`category`/`intent`, 11 categories, 27 intents (`track_order`, `get_refund`, `create_account`, etc.). It does **not** contain internship questions, and its intent names were never renamed or repurposed as internship intents.

It's still used here because a from-scratch NLP pipeline needs a real, sizeable dataset to demonstrate preprocessing, TF-IDF, and Logistic Regression training/evaluation — exactly what an internship mentor would want to see built and explained. The internship-specific behavior of the app, though, comes entirely from the separate FAQ dataset described next.

## 3. Kaggle Data vs. Internship FAQ Data

| | Kaggle dataset | Internship FAQ dataset |
|---|---|---|
| Domain | `customer_service` | `internship` |
| Source | `bitext` | `custom_faq` |
| Rows | 21,534 | 20 intents × 6–10 paraphrased questions ≈ 172 rows |
| Intents | 27 (track_order, get_refund, ...) | 20 (eligibility, stipend, ...) |
| Purpose in this app | Demonstrates the ML pipeline; also lets the app recognize when a question is **not** about the internship (so it can hand it to Gemini as a general question instead of forcing an internship label) | Provides the actual internship intents + the safe FAQ context sent to Gemini |

Both are combined into **one** training set with `text`, `intent`, `domain`, `source` columns (see Section 6 for why one unified classifier was used instead of two separate models), but they are never mixed at the *label* level — a `track_order` row is never relabeled as an internship intent or vice versa.

## 4. Internship FAQ Domain

20 internship intents, defined in `data/internship_faq/internship_faq.json`:

`eligibility`, `application_process`, `application_deadline`, `internship_duration`, `required_documents`, `internship_fee`, `stipend`, `mentor_information`, `project_submission`, `project_requirements`, `evaluation`, `certificate`, `attendance`, `work_hours`, `internship_extension`, `contact_coordinator`, `onboarding`, `internship_completion`, `leave_policy`, `technical_support`

Each FAQ entry has an `intent`, several paraphrased `questions` (used as training examples), and a generic, **non-fabricated** `answer` — e.g. for `internship_duration`:

> "The exact internship duration is not available in the current internship FAQ. Internship durations typically range from a few weeks to a few months depending on the program. Please check the official internship notice or contact the internship coordinator for the specific duration of your program."

No real deadlines, fees, stipend amounts, mentor names, or contact details are invented anywhere in this file — that's intentional, and you're expected to edit these answers with your actual organization's information later. The JSON structure is deliberately simple (one object per intent) so it's easy to edit.

## 5. NLP Preprocessing (`backend/preprocess.py`)

Same pipeline applied identically at training and inference time:
1. Lowercase
2. Remove unnecessary punctuation (keeps letters, digits, apostrophes)
3. Tokenize (NLTK `word_tokenize`)
4. Remove stopwords (NLTK English list)
5. Lemmatize (NLTK `WordNetLemmatizer`)

Negation words (`not`, `no`, `cannot`, `never`, `don't`, etc.) are **excluded** from stopword removal, since they flip the meaning of queries like *"I cannot access the portal"*.

## 6. Why One Unified Classifier (Not Two Separate Models)

Two designs were considered:
- **(a) Two separate classifiers** — one for customer-service intents, one for internship intents.
- **(b) One classifier trained on the combined dataset**, with `domain`/`source` columns tracking where each row came from.

**Option (b) was implemented.** It's simpler to build, train, save, and explain to a mentor (one vectorizer, one classifier, one training script), and it still lets the app tell internship questions apart from everything else: `backend/train.py` saves an `intent → domain` lookup (`models/intent_domain_map.json`) built directly from the actual training data, and `backend/chatbot.py` uses it to decide whether a predicted intent belongs to the internship domain.

## 7. TF-IDF + Logistic Regression

`TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.9, sublinear_tf=True)` — `min_df=1` (instead of the usual 2+) is used specifically so the small internship intents (6–10 examples each) aren't pruned out of the vocabulary.

`LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced", random_state=42)` — `class_weight="balanced"` helps the model pay attention to the much smaller internship classes against the much larger Kaggle classes (`payment_issue` alone has 4,365 examples).

## 8. Intent Classification & Confidence Threshold

`classifier.predict_proba()` gives a probability per intent; the highest is the prediction's confidence. Threshold is configurable via `.env` (`CONFIDENCE_THRESHOLD`), default **0.60**.

**Decision logic** (`backend/chatbot.py`):
```
predict intent + confidence
if confidence >= threshold AND intent's domain == "internship":
    -> answer using that intent's FAQ context (sent to Gemini)
else:
    -> intent reported as "general"
       (covers: low confidence, OR a confident match against one of the
       Kaggle customer-service intents, which this app does not serve)
    -> Gemini answers the raw question directly, no FAQ context
```

This is why a confidently-classified `track_order` question doesn't get treated as an internship topic — it's a real, confident prediction, but its domain isn't internship, so it's routed to Gemini as a general question instead.

## 9. Evaluation Results (actual — from `evaluation/evaluate.py`)

Held-out stratified test split (`test_size=0.2`, `random_state=42`) over the **combined** 47-intent dataset (27 customer-service + 20 internship), test set size = **4,332**.

| Metric (all 47 intents) | Score |
|---|---|
| Accuracy | **0.9885** |
| Precision (macro) | 0.8041 |
| Recall (macro) | 0.8145 |
| F1-score (macro) | 0.7971 |
| Precision (weighted) | 0.9887 |
| Recall (weighted) | 0.9885 |
| F1-score (weighted) | 0.9883 |

**Honest caveat:** the macro-average and internship-only numbers are pulled down by how small the internship dataset is. On the **internship-only subset of the test split (28 examples total across 20 classes — about 1 per class)**, accuracy was **0.57**. With only 1–2 test examples per internship class, a single misclassification swings that class's score from 0% to 100%, so this number is statistically noisy rather than a reliable estimate — see Section 14 (Limitations) and the qualitative check below for a fuller picture.

**Qualitative check (paraphrased questions not seen in training, run manually against the trained model):**

| Question | Predicted intent | Confidence | Routed as |
|---|---|---|---|
| "What is the internship duration?" | internship_duration | 0.985 | internship ✅ |
| "What documents are required?" | required_documents | 0.997 | internship ✅ |
| "What are the eligibility criteria?" | eligibility | 0.939 | internship ✅ |
| "How is the internship evaluated?" | evaluation | 0.985 | internship ✅ |
| "Will I receive a certificate?" | certificate | 0.998 | internship ✅ |
| "Can I take leave?" | leave_policy | 0.853 | internship ✅ |
| "How do I contact my mentor?" | mentor_information | 0.993 | internship ✅ |
| "Is there a stipend?" | stipend | 0.996 | internship ✅ |
| "How do I apply?" | application_process | 0.545 | general (below threshold) ⚠️ |
| "How do I submit my project?" | project_requirements | 0.504 | general (below threshold) ⚠️ |
| "What is Python?" | track_order | 0.051 | general ✅ |
| "What is the weather today?" | track_order | 0.051 | general ✅ |

Most internship questions are classified correctly with high confidence; a couple of short/ambiguous phrasings ("How do I apply?") land just under the 0.60 threshold and get treated as general questions instead — the assistant still answers them via Gemini, just without FAQ-specific grounding. General/off-domain questions consistently score very low confidence, exactly as intended.

Full per-class report and the 47×47 confusion matrix are saved to `evaluation/classification_report.txt` and `evaluation/confusion_matrix.png` — **regenerate both after any change to the data or the internship FAQ file, and update this section with your own numbers.**

## 10. Gemini Integration (`backend/gemini_service.py`)

Uses the current official `google-genai` Python SDK. Gemini receives the user's question, the detected intent, confidence, domain, and (for internship matches) the FAQ context — and is explicitly instructed not to invent policies, deadlines, fees, stipend amounts, mentor names, or contact details beyond what's in the FAQ context, and not to fabricate current/real-time data (weather, stock prices, live scores). If the Gemini call itself fails (bad key, network issue, quota), the backend still returns a safe, non-crashing response.

## 11. General Question Handling

Anything that isn't a confident internship match — a general-knowledge question, a joke request, math, or even a confidently-recognized customer-service intent from the Kaggle side — is labeled `"general"` and sent to Gemini with no FAQ context, so Gemini answers it using its own knowledge instead of forcing it into an internship category.

## 12. FastAPI Backend

- `GET /` — health/status check
- `POST /chat`

  Request:
  ```json
  { "message": "How long is the internship?" }
  ```

  Response:
  ```json
  {
    "response": "...",
    "intent": "internship_duration",
    "confidence": 0.98,
    "domain": "internship"
  }
  ```

  For general questions, `intent` and `domain` are `"general"`.

- Interactive Swagger docs at `/docs`, CORS enabled for the static frontend.

## 13. Frontend

Plain HTML/CSS/JavaScript (`frontend/`), titled **"Internship FAQ & AI Assistant"** with the subtitle **"Internship FAQ • NLP Intent Matching • Gemini AI"**. Chat bubbles, Enter-to-send, loading indicator, error banner, and a collapsible technical/debug panel showing intent, confidence, and domain. Uses native `fetch()` only.

## 14. Project Structure

```
internship-faq-chatbot/
│
├── data/
│   ├── raw/
│   │   └── 20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv
│   ├── internship_faq/
│   │   └── internship_faq.json
│   └── processed/
│       └── combined_training_dataset.csv   (generated by train.py)
│
├── models/
│   ├── vectorizer.pkl
│   ├── intent_classifier.pkl
│   ├── label_encoder.pkl
│   └── intent_domain_map.json              (generated by train.py)
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── chatbot.py
│   ├── preprocess.py
│   ├── train.py
│   ├── gemini_service.py
│   ├── config.py
│   └── utils.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── evaluation/
│   ├── evaluate.py
│   ├── classification_report.txt   (generated)
│   └── confusion_matrix.png        (generated)
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 15. Installation (Windows 11 / PowerShell / VS Code, Python 3.12)

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

Then open `.env` and set your real key:
```
GEMINI_API_KEY=your_actual_key_here
```

## 16. Training

```powershell
python -m backend.train
```

Loads the Kaggle CSV + expands the internship FAQ JSON into training rows, combines them, validates and preprocesses the combined dataset, splits it (stratified, `random_state=42`), fits TF-IDF, trains Logistic Regression, prints a full test-set report, and saves `models/vectorizer.pkl`, `models/intent_classifier.pkl`, `models/label_encoder.pkl`, and `models/intent_domain_map.json`.

## 17. Evaluation

```powershell
python -m evaluation.evaluate
```

Recreates the exact same test split and prints/saves the metrics in Section 9, plus the confusion matrix image.

## 18. Running

**Terminal 1:**
```powershell
uvicorn backend.main:app --reload
```

**Frontend:** double-click `frontend/index.html`, or use VS Code's Live Server extension.

## 19. API Testing Examples

```powershell
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{\"message\": \"What is the internship duration?\"}'
```

Swagger UI: `http://127.0.0.1:8000/docs`

## 20. Test Questions

**Internship questions:**
1. "What is the internship duration?"
2. "What documents are required?"
3. "How do I apply?"
4. "What are the eligibility criteria?"
5. "How is the internship evaluated?"
6. "Will I receive a certificate?"
7. "Can I take leave?"
8. "How do I contact my mentor?"
9. "How do I submit my project?"
10. "Is there a stipend?"

**General questions:**
11. "What is Python?"
12. "What is machine learning?"
13. "Explain API."
14. "What is FastAPI?"
15. "Tell me a joke."
16. "What is the capital of India?"
17. "What is 25 multiplied by 4?"
18. "What is artificial intelligence?"

**Out-of-domain / current-data questions (Gemini should not fabricate live data):**
19. "What is the weather today?"
20. "What is the current stock price?"

## 21. Limitations

- **Small internship dataset.** 20 intents with only 6–10 paraphrased examples each is enough for the classifier to recognize clearly-worded questions with high confidence (see Section 9's qualitative table), but it's too little data for the held-out test metric to be statistically meaningful (1–2 test examples per class), and short/ambiguous phrasings can fall under the confidence threshold. Adding more paraphrases per intent is the single highest-impact improvement available (see Section 22).
- **Class imbalance.** The Kaggle classes vastly outnumber the internship classes (`payment_issue`: 4,365 examples vs. `certificate`: ~7). `class_weight="balanced"` helps, but doesn't fully offset a 500:1 ratio.
- **TF-IDF has no semantic understanding.** It matches vocabulary/n-grams, not meaning — truly novel phrasings with none of the training vocabulary can be missed.
- **FAQ answers are intentionally generic.** No real deadlines, fees, or contact details are included; you must edit `data/internship_faq/internship_faq.json` with your organization's actual information before real use.
- **Gemini dependency.** If the API is unreachable, responses fall back to a simple templated message rather than a fully generated one.

## 22. Future Improvements

- Expand each internship intent to 20–30 paraphrased examples for a statistically meaningful evaluation and better recall on short/ambiguous questions
- Fill in real, organization-specific answers in `internship_faq.json`
- Add a validation split / cross-validation for more robust threshold tuning
- Add multi-turn conversation context to the Gemini prompt
- Add automated tests for `backend/chatbot.py` and `backend/preprocess.py`
- Consider a lightweight synonym/paraphrase augmentation step for the internship training data
