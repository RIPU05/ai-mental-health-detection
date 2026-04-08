"""
Train a multi-class mental health classifier (TF-IDF + LogisticRegression).

Input:  data/processed/combined_dataset.csv  (text, label)
Output: models/mental_health_model.pkl        (sklearn Pipeline)

Labels: depression, anxiety, anger, happy, normal
        (+ any additional labels present in the CSV)

The Pipeline uses LogisticRegression (multi_class='auto') so that
predict_proba() is available for confidence scores in the UI.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-class mental health classifier."
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/combined_dataset.csv",
        help="Path to the combined CSV dataset (text, label columns).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples required per class (classes below this are dropped).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Run first:  python scripts/build_combined_dataset.py"
        )

    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path, usecols=["text", "label"])
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(preprocess_text)
    df = df[df["text"].str.strip().str.len() > 5]

    # Drop classes with too few samples to stratify-split.
    class_counts = df["label"].value_counts()
    valid_classes = class_counts[class_counts >= args.min_samples].index
    dropped = set(df["label"].unique()) - set(valid_classes)
    if dropped:
        print(f"WARNING: dropping under-represented classes: {sorted(dropped)}")
    df = df[df["label"].isin(valid_classes)].reset_index(drop=True)

    print("\nClass distribution:")
    for label, cnt in df["label"].value_counts().items():
        print(f"  {label:15s}: {cnt:,}")
    print(f"  {'TOTAL':15s}: {len(df):,}\n")

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF + Logistic Regression pipeline.
    # LogisticRegression is chosen over LinearSVC because it exposes
    # predict_proba(), which the UI uses for confidence scores.
    pipeline: Pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    print("Training...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nEvaluation on held-out test set:")
    print(classification_report(y_test, y_pred))

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "mental_health_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Saved model → {model_path}")
    print(f"Classes: {list(pipeline.classes_)}")


if __name__ == "__main__":
    main()
