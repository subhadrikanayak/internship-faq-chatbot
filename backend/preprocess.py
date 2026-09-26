"""
backend/preprocess.py

Dedicated text preprocessing module for the intent classification pipeline.

Pipeline steps:
    1. Lowercasing
    2. Punctuation / unnecessary character removal
    3. Tokenization (NLTK)
    4. Stopword removal (NLTK stopwords, with negation words preserved)
    5. Lemmatization (NLTK WordNetLemmatizer)

IMPORTANT:
Negation words such as "not", "no", "cannot", "n't" etc. are intentionally
KEPT even though they normally appear in the standard NLTK stopword list.
Losing them would flip the meaning of queries such as "I cannot log in"
vs "I can log in" - this matters for both the customer-service intents
and internship questions like "I cannot access the portal".
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:
    """
    Make sure all NLTK resources required by this module are available.
    Downloads them quietly if missing. Safe to call multiple times.
    """
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()

# Words that must NEVER be removed, even though NLTK's default stopword
# list would normally strip them. These carry meaning-critical negation.
NEGATION_WORDS = {
    "no",
    "nor",
    "not",
    "don",
    "don't",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
    "cannot",
    "can't",
    "cant",
    "never",
    "without",
}

_BASE_STOPWORDS = set(stopwords.words("english"))
STOPWORDS = _BASE_STOPWORDS - NEGATION_WORDS

_lemmatizer = WordNetLemmatizer()

_PUNCT_PATTERN = re.compile(r"[^a-z0-9'\s]")


def clean_text(text: str) -> str:
    """Lowercase and strip unnecessary punctuation while keeping apostrophes."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = text.replace("’", "'")
    text = _PUNCT_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Tokenize cleaned text using NLTK's word_tokenize."""
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """Remove stopwords while preserving negation words."""
    return [t for t in tokens if t not in STOPWORDS and t not in string.punctuation]


def lemmatize(tokens: list) -> list:
    """Lemmatize tokens (verb pass then noun pass)."""
    lemmatized = []
    for tok in tokens:
        verb_lemma = _lemmatizer.lemmatize(tok, pos="v")
        noun_lemma = _lemmatizer.lemmatize(verb_lemma, pos="n")
        lemmatized.append(noun_lemma)
    return lemmatized


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline used both at training time and at
    inference time (must stay IDENTICAL in both places).

    Returns a single cleaned, space-joined string ready for TF-IDF.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "What is the internship duration?",
        "I cannot log into the internship portal",
        "How do I apply for this internship?",
    ]
    for s in samples:
        print(f"RAW  : {s}")
        print(f"CLEAN: {preprocess_text(s)}")
        print()
