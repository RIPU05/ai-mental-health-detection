from __future__ import annotations

import argparse
import os
import pickle
import re
import string
from functools import lru_cache
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def _ensure_nltk() -> None:
    # Keep downloads minimal; we avoid Punkt tokenizers by using regex tokenization below.
    for pkg in ("stopwords", "wordnet"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg)


@lru_cache(maxsize=1)
def _resources() -> tuple[set[str], WordNetLemmatizer]:
    _ensure_nltk()
    return set(stopwords.words("english")), WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    stop_words, lemmatizer = _resources()

    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.findall(r"[a-z']+", text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def _detect_default_dataset() -> str | None:
    candidates = [
        Path("data/raw/archive (3)/depression_dataset_reddit_cleaned.csv"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    # Fall back: first CSV found under data/raw
    raw = Path("data/raw")
    if raw.exists():
        found = sorted(raw.rglob("*.csv"))
        if found:
            return str(found[0])
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline emotion model (TF-IDF + LogisticRegression).")
    parser.add_argument(
        "--data-path",
        default=_detect_default_dataset(),
        help="Path to CSV containing 'text' and 'label' columns.",
    )
    parser.add_argument(
        "--text-col",
        default=None,
        help="Name of the text column (auto-detected if omitted).",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Name of the label column (auto-detected if omitted).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_path = args.data_path

    if not data_path:
        raise FileNotFoundError(
            "No dataset path provided and none could be auto-detected. "
            "Pass one with: python scripts/train_emotion_model.py --data-path <path-to-csv>"
        )
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}\n"
            "Tip: your repo currently contains: data/raw/archive (3)/depression_dataset_reddit_cleaned.csv"
        )

    df = pd.read_csv(data_path)
    print("Columns in dataset:", df.columns.tolist())

    # Initialize NLTK resources once (avoid repeated checks during pandas apply).
    _resources()

    # Auto-detect common column names if the user didn't specify them.
    text_col = args.text_col
    label_col = args.label_col

    if text_col is None:
        for candidate in ("text", "clean_text", "content", "sentence", "utterance"):
            if candidate in df.columns:
                text_col = candidate
                break
    if label_col is None:
        for candidate in ("label", "target", "class", "is_depression", "emotion"):
            if candidate in df.columns:
                label_col = candidate
                break

    if not text_col or text_col not in df.columns or not label_col or label_col not in df.columns:
        raise ValueError(
            "Could not find required columns.\n"
            f"Detected text_col={text_col!r}, label_col={label_col!r}.\n"
            "Provide them explicitly, e.g.:\n"
            "  python scripts/train_emotion_model.py --data-path <csv> --text-col clean_text --label-col is_depression"
        )

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].apply(preprocess_text)

    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)

    with open("models/emotion_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/emotion_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Saved model to models/emotion_model.pkl")
    print("Saved vectorizer to models/emotion_vectorizer.pkl")


if __name__ == "__main__":
    main()