"""
Train a multi-class mental health classifier (TF-IDF + LogisticRegression).

Primary dataset : attached_assets/Combined_Data_1775671429588.csv
                  columns: statement (text), status (label)
                  classes: Normal, Depression, Suicidal, Anxiety,
                           Bipolar, Stress, Personality disorder

Supplementary   : attached_assets/stressed_anxious_cleaned_1775671387357.csv
                  columns: Text, is_stressed/anxious
                  only rows where is_stressed/anxious == 1 are used → label "stress"
                  (adds ~3 k samples to the under-represented Stress class)

Skipped         : mental_heath_unbanlanced.csv      (subset of Combined_Data)
                  mental_heath_feature_engineered.csv (same data + numeric features
                  we do not need — TF-IDF learns its own features)

Output          : models/mental_health_model.pkl   (sklearn Pipeline)

The Pipeline wraps TF-IDF + LogisticRegression so predict_proba() is available
for confidence scores in the Streamlit UI.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
ASSETS_DIR   = Path("attached_assets")
PRIMARY_CSV  = ASSETS_DIR / "Combined_Data_1775671429588.csv"
STRESS_CSV   = ASSETS_DIR / "stressed_anxious_cleaned_1775671387357.csv"
MODEL_OUT    = Path("models") / "mental_health_model.pkl"

# Minimum samples per class to be included in training
MIN_SAMPLES = 200


# ── Label normalisation map ────────────────────────────────────────────────────
# Keys are raw values from the CSV (after .strip()).
# Values are the canonical lowercase labels stored in the model.
LABEL_MAP: dict[str, str] = {
    "Normal":               "normal",
    "Depression":           "depression",
    "Suicidal":             "suicidal",
    "Anxiety":              "anxiety",
    "Bipolar":              "bipolar",
    "Stress":               "stress",
    "Personality disorder": "personality disorder",
}


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_primary() -> pd.DataFrame:
    """Load Combined_Data.csv → normalised (text, label) frame."""
    print(f"Loading primary dataset: {PRIMARY_CSV}")
    df = pd.read_csv(PRIMARY_CSV, usecols=["statement", "status"])
    df = df.rename(columns={"statement": "text", "status": "label"})
    df["label"] = df["label"].map(LABEL_MAP)
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with unknown labels.")
    print(f"  Loaded {len(df):,} rows from primary dataset.")
    return df


def load_stress_supplement() -> pd.DataFrame:
    """Load stressed_anxious_cleaned.csv; keep only positive (stressed) rows."""
    print(f"Loading stress supplement: {STRESS_CSV}")
    df = pd.read_csv(STRESS_CSV)
    col = "is_stressed/anxious"
    if col not in df.columns or "Text" not in df.columns:
        print("  WARNING: unexpected columns — skipping stress supplement.")
        return pd.DataFrame(columns=["text", "label"])
    df = df[df[col] == 1][["Text"]].copy()
    df = df.rename(columns={"Text": "text"})
    df["label"] = "stress"
    print(f"  Loaded {len(df):,} stress/anxious supplement rows.")
    return df


def main() -> None:
    # ── Load & merge ──────────────────────────────────────────────────────────
    primary  = load_primary()
    stress   = load_stress_supplement()
    df = pd.concat([primary, stress], ignore_index=True)

    # ── Clean ────────────────────────────────────────────────────────────────
    df["text"] = df["text"].apply(preprocess_text)
    df = df[df["text"].str.len() > 5].drop_duplicates(subset="text")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    # ── Drop tiny classes ────────────────────────────────────────────────────
    class_counts = df["label"].value_counts()
    valid = class_counts[class_counts >= MIN_SAMPLES].index
    dropped_classes = sorted(set(df["label"]) - set(valid))
    if dropped_classes:
        print(f"\nWARNING: dropping under-represented classes: {dropped_classes}")
    df = df[df["label"].isin(valid)].reset_index(drop=True)

    # ── Class distribution ───────────────────────────────────────────────────
    print("\nFinal class distribution:")
    for label, cnt in df["label"].value_counts().items():
        print(f"  {label:22s}: {cnt:,}")
    print(f"  {'TOTAL':22s}: {len(df):,}\n")

    X, y = df["text"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Model ────────────────────────────────────────────────────────────────
    pipeline: Pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
        )),
        ("clf", LogisticRegression(
            max_iter=5000,
            random_state=42,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
        )),
    ])

    print("Training…")
    pipeline.fit(X_train, y_train)

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    print("\nEvaluation on held-out test set (20%):")
    print(classification_report(y_test, y_pred))

    # ── Save ─────────────────────────────────────────────────────────────────
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Saved model → {MODEL_OUT}")
    print(f"Classes    : {list(pipeline.classes_)}")


if __name__ == "__main__":
    main()
