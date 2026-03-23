from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

import streamlit as st

from ai_mh_detection.config import AppConfig, default_config_path
from ai_mh_detection.preprocessing import TextPreprocessor

try:
    # `audio/` lives at repo root, so it should be importable when running from the project folder.
    from audio.speech_to_text import speech_to_text
except Exception:  # pragma: no cover
    speech_to_text = None  # type: ignore[assignment]


def _load_config() -> AppConfig:
    try:
        return AppConfig.load(default_config_path())
    except Exception:
        return AppConfig(data={})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@st.cache_resource
def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _get_recommendation(is_depression: int) -> str:
    if is_depression == 1:
        return (
            "Consider reaching out to a mental health professional, talking to someone you trust, "
            "and maintaining a regular routine (sleep, meals, movement). If you feel unsafe, "
            "contact local emergency or crisis support immediately."
        )
    return (
        "No depression signal detected by this model. Keep tracking your mood, maintain healthy habits, "
        "and seek support early if symptoms increase."
    )


def _chatbot_reply(emotion_label: str, mental_pred: int, user_text: str | None = None) -> str:
    """
    Simple supportive (non-clinical) response template based on detected emotion.
    """
    emotion_label_norm = str(emotion_label).lower()
    context = f"You mentioned {emotion_label!s} feelings." if user_text else f"Detected emotion: {emotion_label!s}."

    if mental_pred == 1:
        safety_hint = (
            "If you feel unsafe or at risk of self-harm, please contact local emergency services or a crisis hotline right now."
        )
    else:
        safety_hint = "If things feel overwhelming, it can still help to talk with a mental health professional or someone you trust."

    if "negative" in emotion_label_norm or "sad" in emotion_label_norm or "depressed" in emotion_label_norm:
        return (
            f"{context}\n\n"
            "That sounds really heavy. Would you be open to sharing what part of today has felt the hardest?\n\n"
            f"{safety_hint}"
        )
    if "positive" in emotion_label_norm or "happy" in emotion_label_norm or "relieved" in emotion_label_norm or "joy" in emotion_label_norm:
        return (
            f"{context}\n\n"
            "I’m glad you’re feeling some relief. What helped most—something specific that happened, or a shift in how you’re thinking?\n\n"
            f"{safety_hint}"
        )

    # Neutral / unknown
    return (
        f"{context}\n\n"
        "Thanks for sharing. When you think about the last few hours, what feeling comes up most often—anxious, sad, tired, or something else?"
        f"\n\n{safety_hint}"
    )


def main() -> None:
    cfg = _load_config().data
    st.set_page_config(
        page_title=cfg.get("app", {}).get("title", "AI Mental Health Detection"),
        layout="centered",
    )
    st.title(cfg.get("app", {}).get("title", "AI Mental Health Detection"))
    st.caption("Text-based emotion and mental health screening demo")

    model_dir = _repo_root() / "models"
    emotion_model_path = model_dir / "emotion_model.pkl"
    emotion_vectorizer_path = model_dir / "emotion_vectorizer.pkl"
    mental_health_model_path = model_dir / "mental_health_model.pkl"

    st.header("Model Status")
    status_cols = st.columns(3)
    status_cols[0].metric("Emotion model", "Loaded" if emotion_model_path.exists() else "Missing")
    status_cols[1].metric("Emotion vectorizer", "Loaded" if emotion_vectorizer_path.exists() else "Missing")
    status_cols[2].metric("Mental health model", "Loaded" if mental_health_model_path.exists() else "Missing")

    if not (emotion_model_path.exists() and emotion_vectorizer_path.exists() and mental_health_model_path.exists()):
        st.error("One or more model files are missing under `models/`. Train models first, then reload.")
        st.info(
            "Expected files: `models/emotion_model.pkl`, `models/emotion_vectorizer.pkl`, "
            "`models/mental_health_model.pkl`"
        )
        return

    emotion_model = _load_pickle(emotion_model_path)
    emotion_vectorizer = _load_pickle(emotion_vectorizer_path)
    mental_health_model = _load_pickle(mental_health_model_path)
    mental_health_vectorizer_path = model_dir / "mental_health_vectorizer.pkl"
    pre = TextPreprocessor()

    st.header("Input")
    if "mood_history" not in st.session_state:
        st.session_state.mood_history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "last_emotion" not in st.session_state:
        st.session_state.last_emotion = None
    if "last_mental_pred" not in st.session_state:
        st.session_state.last_mental_pred = None

    tabs = st.tabs(["Analyze", "Mood History", "Chatbot"])

    with tabs[0]:
        st.subheader("Input")
        text = st.text_area(
            "Enter text (optional if you upload audio)",
            height=160,
            placeholder="Type a journal entry, chat text, or statement...",
        )
        audio_file = st.file_uploader("Upload audio (wav, optional)", type=["wav"])
        run = st.button("Analyze", type="primary", use_container_width=True)

        if not run:
            st.caption("Enter text and click Analyze.")
            return

        try:
            # Prefer audio transcription if audio is provided; otherwise use the text input.
            if audio_file is not None:
                if speech_to_text is None:
                    st.error(
                        "Speech transcription is unavailable. Install dependency `SpeechRecognition` "
                        "and restart the app."
                    )
                    st.stop()

                # Save uploaded audio to a temporary file for SpeechRecognition's AudioFile.
                model_dir = _repo_root() / "data" / "processed"
                model_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = model_dir / f"uploaded_audio_{int(datetime.now().timestamp())}.wav"
                tmp_path.write_bytes(audio_file.getbuffer())

                st.info("Transcribing audio...")
                transcribed_text = speech_to_text(str(tmp_path))
                if not transcribed_text or not str(transcribed_text).strip():
                    st.warning("Transcription returned empty text.")
                    return

                st.subheader("Converted Text (from audio)")
                st.write(transcribed_text)
                input_text = transcribed_text
            else:
                input_text = text or ""

            if not input_text or not str(input_text).strip():
                st.warning("Please enter text or upload a non-empty audio file.")
                return

            processed_text = pre.transform(input_text)
            if not processed_text:
                st.warning("After preprocessing, no usable text remains.")
                return

            emotion_features = emotion_vectorizer.transform([processed_text])
            emotion_pred = emotion_model.predict(emotion_features)[0]

            # Mental health model can be either:
            # 1) a sklearn Pipeline (tfidf -> SVM), which accepts raw text, or
            # 2) a standalone SVM that requires TF-IDF features from a saved vectorizer.
            if isinstance(mental_health_model, Pipeline):
                mental_pred_raw = mental_health_model.predict([processed_text])[0]
            else:
                if not mental_health_vectorizer_path.exists():
                    raise ValueError(
                        "mental_health_model.pkl is not a Pipeline, but mental_health_vectorizer.pkl is missing. "
                        "Train again or add the missing vectorizer."
                    )
                mental_vectorizer = _load_pickle(mental_health_vectorizer_path)
                mental_features = mental_vectorizer.transform([processed_text])
                mental_pred_raw = mental_health_model.predict(mental_features)[0]

            try:
                mental_pred = int(mental_pred_raw)
                mental_label = "Depression" if mental_pred == 1 else "Not Depression"
            except (TypeError, ValueError):
                mental_label = (
                    "Depression" if str(mental_pred_raw).lower() in {"1", "true", "depression"} else "Not Depression"
                )
                mental_pred = 1 if mental_label == "Depression" else 0

            st.success("Analysis complete.")

            st.subheader("Detected Emotion")
            st.write(str(emotion_pred))

            st.subheader("Mental Health Prediction")
            if mental_pred == 1:
                st.error(mental_label)
            else:
                st.success(mental_label)

            st.subheader("Recommendation")
            st.info(_get_recommendation(mental_pred))

            # Update session state
            st.session_state.last_emotion = emotion_pred
            st.session_state.last_mental_pred = mental_pred
            st.session_state.mood_history.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "emotion": str(emotion_pred),
                    "prediction": "Depression" if mental_pred == 1 else "Not Depression",
                }
            )

            # Seed chatbot with an emotion-based assistant message
            assistant_msg = _chatbot_reply(str(emotion_pred), mental_pred, user_text=None)
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_msg})
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    with tabs[1]:
        st.subheader("Mood History Tracker")
        history: list[dict[str, str]] = st.session_state.mood_history
        if not history:
            st.caption("No history yet. Run an analysis first.")
        else:
            st.dataframe(history[-20:], use_container_width=True)
            depression_count = sum(1 for r in history if r.get("prediction") == "Depression")
            st.metric("Depression flagged (in last runs)", depression_count)
            if st.button("Clear history", use_container_width=True):
                st.session_state.mood_history = []
                st.success("History cleared.")

    with tabs[2]:
        st.subheader("Support Chatbot")
        if not st.session_state.chat_messages:
            st.caption("Run an analysis first to enable emotion-based support messages.")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_chat = st.chat_input("Ask a supportive question (uses your last detected emotion)...")
        if user_chat:
            st.session_state.chat_messages.append({"role": "user", "content": user_chat})

            last_emotion = st.session_state.last_emotion
            last_mental_pred = st.session_state.last_mental_pred
            if last_emotion is None or last_mental_pred is None:
                reply = "Please run an analysis first so I can respond based on the detected emotion."
            else:
                reply = _chatbot_reply(str(last_emotion), int(last_mental_pred), user_text=user_chat)

            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)


if __name__ == "__main__":
    main()
