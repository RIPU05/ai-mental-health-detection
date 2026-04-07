# AI Mental Health Detection

## Overview
A Python-based Streamlit dashboard for AI-driven mental health screening. The app analyzes text (or transcribed audio) to detect emotions and predict depression risk, then provides supportive chatbot responses and actionable recommendations.

## Architecture
- **Framework**: Streamlit (Python)
- **Entry point**: `streamlit_app.py`
- **Source package**: `src/ai_mh_detection/`
- **Config**: `configs/default.yaml`
- **Models**: Pre-trained `.pkl` files in `models/`

## Key Modules
- `src/ai_mh_detection/dashboard/app.py` — Main Streamlit UI with Analyze, Mood History, and Chatbot tabs
- `src/ai_mh_detection/preprocessing/` — Text cleaning and normalization
- `src/ai_mh_detection/config.py` — App configuration via Pydantic
- `audio/speech_to_text.py` — Audio transcription via SpeechRecognition
- `scripts/` — Model training scripts

## Models
- `models/emotion_model.pkl` + `models/emotion_vectorizer.pkl` — Emotion classification (positive/negative/neutral)
- `models/mental_health_model.pkl` + `models/mental_health_vectorizer.pkl` — Depression risk prediction

## Configuration
- Streamlit server config: `.streamlit/config.toml` (port 5000, host 0.0.0.0, CORS disabled)
- App config: `configs/default.yaml`

## Running
- Workflow: `streamlit run streamlit_app.py`
- Port: 5000

## Dependencies
All managed via `requirements.txt` and `pyproject.toml`. Installed into `.pythonlibs/`.
- Core: numpy, pandas, scikit-learn, nltk, streamlit, pydantic, PyYAML
- Audio: SpeechRecognition, pydub, librosa, soundfile
- Optional: torch, transformers (for deep learning features)
- Optional: OPENAI_API_KEY env var for enhanced chatbot responses

## Notes
- The app works fully offline (rule-based chatbot fallback, no API key required)
- Setting `OPENAI_API_KEY` enables GPT-powered chatbot responses
- Audio transcription requires ffmpeg system dependency for MP3 support
