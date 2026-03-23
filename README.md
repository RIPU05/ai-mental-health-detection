# AI Mental Health Detection

Python project scaffold for an AI-based mental health detection system with:

- Text preprocessing
- Audio processing
- Emotion detection
- Mental health prediction
- Chatbot
- Recommendation system
- Streamlit dashboard

## Quick start

Create an environment and install:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -e .
```

Run the dashboard:

```bash
streamlit run streamlit_app.py
```

## Project structure

```
ai-mental-health-detection/
  configs/
    default.yaml
  data/
    raw/
    processed/
  models/
  notebooks/
  scripts/
  src/
    ai_mh_detection/
      __init__.py
      config.py
      preprocessing/
        __init__.py
        text_preprocessor.py
      audio/
        __init__.py
        audio_processor.py
      emotion/
        __init__.py
        emotion_detector.py
      prediction/
        __init__.py
        mental_health_predictor.py
      chatbot/
        __init__.py
        chatbot.py
      recommendation/
        __init__.py
        recommender.py
      dashboard/
        __init__.py
        app.py
      utils/
        __init__.py
        io.py
  tests/
```

## Notes

- The current implementations are **safe, minimal stubs** (rule-based placeholders) meant to be replaced with trained models.
- Put trained artifacts (e.g., `.pkl`, `.pt`) in `models/` and load them inside `emotion/` and `prediction/`.
