from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    # Keep only letters and apostrophes; collapse whitespace.
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SVM mental health classifier (TF-IDF + LinearSVC).")
    parser.add_argument(
        "--data-path",
        default="data/raw/archive (3)/depression_dataset_reddit_cleaned.csv",
        help="Path to the CSV dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_path = args.data_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    required = {"clean_text", "is_depression"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {sorted(required)}. Found: {df.columns.tolist()}")

    df = df[list(required)].dropna()
    df["clean_text"] = df["clean_text"].apply(preprocess_text)
    df["is_depression"] = df["is_depression"].astype(int)

    X = df["clean_text"]
    y = df["is_depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline: Pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("svm", LinearSVC(random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/mental_health_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("Saved model to models/mental_health_model.pkl")


if __name__ == "__main__":
    main()