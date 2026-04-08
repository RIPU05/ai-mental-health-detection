# AI Mental Health Detection

A Streamlit application that analyses text (or transcribed audio) to detect emotional tone and predict mental health conditions, then provides a supportive chatbot and actionable recommendations.

> **Disclaimer:** This tool is for informational and supportive screening purposes only. It is not a medical diagnosis and does not replace professional mental health care.

---

## Features

- **Text analysis** — paste a journal entry, social media post, or any statement
- **Audio input** — upload a WAV or MP3 file; the app transcribes it and analyses the transcript
- **Emotion detection** — classifies the emotional tone (positive / negative / neutral)
- **Mental health condition prediction** — multi-class classifier across 7 conditions (see below)
- **Confidence scores** — top prediction confidence with a top-3 breakdown
- **Personalised recommendations** — condition-aware, actionable guidance for each result
- **Crisis safeguard** — prominent warning and crisis-line links when suicidal ideation is detected
- **Supportive chatbot** — condition-aware, varied responses; falls back to local rule-based if no OpenAI key is set
- **Mood history** — tracks every analysis run with timestamps, conditions, and confidence

---

## Supported Conditions

| Internal label | Display name |
|---|---|
| `normal` | No Major Concern |
| `depression` | Depression |
| `suicidal` | Suicidal Ideation |
| `anxiety` | Anxiety |
| `bipolar` | Bipolar-related Signs |
| `stress` | Stress |
| `personality disorder` | Personality-related Distress |

---

## Project Structure

```
.
├── attached_assets/          # Source datasets (used by training script)
├── audio/
│   └── speech_to_text.py     # Google Speech Recognition wrapper
├── configs/
│   └── default.yaml          # App config (title, thresholds)
├── data/
│   └── processed/            # Intermediate data (git-ignored)
├── models/                   # Trained model artifacts (git-ignored)
│   ├── emotion_model.pkl
│   ├── emotion_vectorizer.pkl
│   └── mental_health_model.pkl
├── scripts/
│   ├── train_emotion_model.py
│   └── train_mental_health_model.py
├── src/ai_mh_detection/
│   ├── config.py
│   ├── preprocessing/
│   └── dashboard/
│       └── app.py            # Main Streamlit app
└── streamlit_app.py          # Entry point
```

---

## Setup

```bash
# 1. Install dependencies
pip install -e .

# 2. Train the mental health model (reads from attached_assets/)
python scripts/train_mental_health_model.py

# 3. Train the emotion model (if not already trained)
python scripts/train_emotion_model.py
```

Models are saved to `models/` (git-ignored — train locally or pull from your artifact store).

---

## Running the App

```bash
streamlit run streamlit_app.py
```

The app runs on port 5000 by default (configured in `.streamlit/config.toml`).

---

## Training Details

### Mental health model (`scripts/train_mental_health_model.py`)

**Datasets used:**

| File | Role | Rows used |
|---|---|---|
| `attached_assets/Combined_Data_1775671429588.csv` | Primary (7 classes) | ~52,700 |
| `attached_assets/stressed_anxious_cleaned_1775671387357.csv` | Stress supplement (positive rows only) | ~4,000 |

**Pipeline:** TF-IDF (30k features, bigrams, sublinear TF) + Logistic Regression (balanced class weights, lbfgs)

**Reported metrics (held-out 20% test set):**

| Metric | Value |
|---|---|
| Accuracy | ~75% |
| Macro F1 | ~0.71 |

To retrain:

```bash
python scripts/train_mental_health_model.py
```

---

## Optional: OpenAI Chatbot

Set the environment variable `OPENAI_API_KEY` to enable GPT-powered chatbot responses. Without it, the app uses a built-in rule-based chatbot with rich, condition-aware templates.

---

## Crisis Resources

If you or someone you know is in crisis, please contact a crisis line:

- International directory: https://www.iasp.info/resources/Crisis_Centres/
- US: 988 Suicide & Crisis Lifeline — call or text **988**
- UK: Samaritans — **116 123**
