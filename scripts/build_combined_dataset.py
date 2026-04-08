"""
Build a unified multi-class mental health dataset.

Sources used:
  1. Depression Reddit dataset  → depression / normal
  2. GoEmotions (3 files)       → anxiety, anger, happy, normal, depression

Unified labels produced:
  depression  - depressive text (reddit) + sadness/grief/remorse (GoEmotions)
  anxiety     - nervousness/fear (GoEmotions)
  anger       - anger/annoyance/disgust/disapproval (GoEmotions)
  happy       - joy/amusement/excitement/optimism/... (GoEmotions)
  normal      - neutral/approval/curiosity/... (GoEmotions) + non-depression reddit

NOTE — dataset limitations:
  The following conditions requested in the task have NO available training data
  in this project's raw datasets. Dedicated labelled corpora are required before
  the model can reliably detect them:
    - stress   (overlaps with anger/annoyance here; needs explicit stress corpus)
    - bipolar  (no data — requires clinical bipolar text dataset)
    - PTSD     (no data — requires PTSD-specific dataset)
    - OCD      (no data — requires OCD-specific dataset)
  These are clearly marked here so they can be added later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ── Label mapping: GoEmotions emotion → unified MH label ──────────────────────
GOEMOTIONS_TO_LABEL: dict[str, str] = {
    # Depression / low mood
    "sadness": "depression",
    "grief": "depression",
    "disappointment": "depression",
    "remorse": "depression",
    # Anxiety
    "nervousness": "anxiety",
    "fear": "anxiety",
    # Anger / dysregulation  (maps to 'anger'; rename to 'stress' once
    # a dedicated stress corpus is available)
    "anger": "anger",
    "annoyance": "anger",
    "disgust": "anger",
    "disapproval": "anger",
    # Happy / positive
    "joy": "happy",
    "amusement": "happy",
    "excitement": "happy",
    "optimism": "happy",
    "pride": "happy",
    "relief": "happy",
    "gratitude": "happy",
    "admiration": "happy",
    "love": "happy",
    "caring": "happy",
    # Normal / neutral
    "neutral": "normal",
    "approval": "normal",
    "confusion": "normal",
    "curiosity": "normal",
    "desire": "normal",
    "embarrassment": "normal",
    "realization": "normal",
    "surprise": "normal",
}

# Pick the "most clinically relevant" label when a row has multiple emotions.
LABEL_PRIORITY: list[str] = ["depression", "anxiety", "anger", "happy", "normal"]


def _clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Loader 1: Reddit depression dataset ───────────────────────────────────────

def load_depression_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Depression dataset not found: {path}")

    df = pd.read_csv(path, usecols=["clean_text", "is_depression"])
    df = df.dropna()
    df["label"] = df["is_depression"].astype(int).map({1: "depression", 0: "normal"})
    df = df.rename(columns={"clean_text": "text"})[["text", "label"]]
    print(f"  [depression dataset] {len(df):,} rows loaded from {path.name}")
    return df


# ── Loader 2: GoEmotions ───────────────────────────────────────────────────────

def load_goemotions(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    parts: list[pd.DataFrame] = []
    for f in csv_files:
        parts.append(pd.read_csv(f))
    df = pd.concat(parts, ignore_index=True)
    print(f"  [GoEmotions] {len(df):,} raw rows from {len(csv_files)} files")

    # Drop rows flagged as very unclear.
    if "example_very_unclear" in df.columns:
        df = df[df["example_very_unclear"] != True]  # noqa: E712

    # Keep only emotion columns that appear in our mapping.
    emotion_cols = [c for c in GOEMOTIONS_TO_LABEL if c in df.columns]
    missing = set(GOEMOTIONS_TO_LABEL) - set(emotion_cols)
    if missing:
        print(f"  [GoEmotions] WARNING: columns not found and skipped: {sorted(missing)}")

    text_col = df["text"].astype(str)

    # For each row build a label using priority order.
    # Vectorised: for each priority label collect a mask of rows where at
    # least one of the mapped GoEmotions columns is 1.
    label_series = pd.Series("", index=df.index, dtype=str)

    # Apply in reverse priority so lower-priority labels get overwritten.
    for priority_label in reversed(LABEL_PRIORITY):
        relevant_cols = [
            c for c, lbl in GOEMOTIONS_TO_LABEL.items()
            if lbl == priority_label and c in emotion_cols
        ]
        if not relevant_cols:
            continue
        mask = df[relevant_cols].any(axis=1)
        label_series[mask] = priority_label

    # Keep only rows that received a label.
    valid = label_series != ""
    result = pd.DataFrame({"text": text_col[valid], "label": label_series[valid]})
    print(f"  [GoEmotions] {len(result):,} rows after label assignment")
    return result


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_dataset.csv"

    print("\nStep 1: Loading datasets...")
    dep_df = load_depression_dataset("data/raw/archive (3)/depression_dataset_reddit_cleaned.csv")
    ge_df = load_goemotions("data/raw/archive (2)/data/full_dataset")

    print("\nStep 2: Merging...")
    combined = pd.concat([dep_df, ge_df], ignore_index=True)

    print("\nStep 3: Cleaning...")
    combined["text"] = combined["text"].apply(_clean)
    combined = combined.dropna(subset=["text", "label"])
    combined = combined[combined["text"].str.strip().str.len() > 5]
    combined = combined.drop_duplicates(subset=["text"])
    combined = combined.reset_index(drop=True)

    print("\nLabel distribution:")
    dist = combined["label"].value_counts()
    for label, count in dist.items():
        print(f"  {label:15s}: {count:,}")
    print(f"  {'TOTAL':15s}: {len(combined):,}")

    combined.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 60)
    print("DATASET LIMITATION NOTICE")
    print("=" * 60)
    print("The following conditions have NO training data in this repo:")
    print("  - stress   (needs dedicated stress text dataset)")
    print("  - bipolar  (needs clinical bipolar dataset)")
    print("  - PTSD     (needs PTSD-specific dataset)")
    print("  - OCD      (needs OCD-specific dataset)")
    print("To add these, obtain labelled CSVs and add loaders above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
