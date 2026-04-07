from __future__ import annotations

import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydub import AudioSegment
import requests
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


def _get_recommendation(*, mental_pred: int, emotion_label: str) -> str:
    """
    Return a short, practical recommendation based on:
    - mental_pred: 1 means depression, 0 means not depression
    - emotion_label: detected emotion string (positive/negative/neutral)

    Note: This project currently predicts depression/not depression; anxiety/stress
    are inferred from the detected emotion for recommendation purposes.
    """
    emotion_label_norm = str(emotion_label).strip().lower()

    # Choose a "state" for recommendations.
    if mental_pred == 1:
        state = "depression"
    elif "anxious" in emotion_label_norm or "panic" in emotion_label_norm:
        state = "anxiety"
    elif "negative" in emotion_label_norm or "sad" in emotion_label_norm or "depressed" in emotion_label_norm:
        state = "stress"
    else:
        state = "normal"

    # Emotion-aware, actionable suggestions (keep simple + concrete).
    if state == "depression":
        return (
            "Try a 3-step support plan (10–20 minutes total):\n"
            "1) Message or call one trusted person: “I’m having a rough time today—can you check in?”\n"
            "2) Do one small body reset: water + short walk/stretch.\n"
            "3) Write 3 lines: what I feel, what I need, what’s one tiny next step.\n"
            "If you feel unsafe or at risk of self-harm, contact local emergency/crisis support now."
        )

    if state == "anxiety":
        return (
            "For anxiety right now (2 minutes):\n"
            "1) Breathing: inhale 4s, exhale 6s, repeat 6 times.\n"
            "2) Grounding: name 5 things you see, 4 you feel, 3 you hear.\n"
            "3) One action: choose a “next 10 minutes” task (small and doable).\n"
            "If anxiety keeps returning, consider professional support."
        )

    if state == "stress":
        return (
            "For stress (quick reset + clarity):\n"
            "1) Identify the trigger: what happened right before this feeling?\n"
            "2) Release tension: unclench jaw/shoulders, relax hands, take 5 slow breaths.\n"
            "3) Pick one coping step: reduce stimulation, or do a 10-minute tidy/walk.\n"
            "If symptoms worsen, reach out to someone supportive or a professional."
        )

    # Normal
    if "positive" in emotion_label_norm or "happy" in emotion_label_norm or "joy" in emotion_label_norm:
        return (
            "Enjoy the positive momentum:\n"
            "1) Note what helped (one specific thing).\n"
            "2) Repeat the smallest part of it tomorrow.\n"
            "3) Share it with someone—connection strengthens mood.\n"
            "Keep monitoring how you feel over the next few hours."
        )

    return (
        "Keep things steady:\n"
        "1) Keep a basic routine (sleep, food, water).\n"
        "2) Do one low-effort check-in: “How am I feeling in my body?”\n"
        "3) If things shift downward, seek support early."
    )


def _normalize_emotion_label(emotion_pred: Any, *, processed_text: str | None = None) -> str:
    """
    Normalize emotion output to one of: "positive", "negative", "neutral" (best-effort).

    Your current `emotion_model.pkl` appears to emit numeric class labels (0/1).
    The chatbot/recommendation logic is string-based, so we map:
      - 1 -> "positive"
      - 0 -> "negative"

    If the predicted label is "negative" and `processed_text` is provided,
    we do a lightweight keyword check to optionally map to "anxious" (so
    recommendations can be more specific).
    """
    # Try numeric mapping first.
    try:
        val = int(emotion_pred)
        if val == 1:
            return "positive"
        if val == 0:
            if processed_text:
                t = processed_text.lower()
                anxious_markers = {"anxious", "anxiety", "panic", "worried", "nervous", "overwhelmed"}
                if any(m in t for m in anxious_markers):
                    return "anxious"
            return "negative"
    except Exception:
        pass

    # Fallback to string normalization.
    s = str(emotion_pred).strip().lower()
    if s in {"0", "false"}:
        return "negative"
    if s in {"1", "true"}:
        return "positive"
    return s


def _local_rule_based_response(user_text: str, emotion_label: str, mental_pred: int) -> str:
    """
    Local, no-API chatbot fallback with varied templates.
    Always returns a non-empty supportive response.
    """
    fallback = (
        "Thanks for sharing this. I’m here with you. What feels hardest right now, and what is one small step that might help in the next 10 minutes?"
    )

    try:
        text = (user_text or "").strip()
        text_norm = text.lower()
        emo = (emotion_label or "neutral").strip().lower()
        pred = int(mental_pred)

        # Infer a richer conversation state from emotion + content.
        if pred == 1:
            state = "depression"
        elif "anxious" in emo or "panic" in emo or any(k in text_norm for k in ("anxious", "anxiety", "panic", "worried", "nervous")):
            state = "anxious"
        elif "stress" in emo or any(k in text_norm for k in ("stressed", "overwhelmed", "deadline", "pressure", "burnout")):
            state = "stress"
        elif "positive" in emo or any(k in emo for k in ("happy", "joy", "relieved", "excited")):
            state = "positive"
        elif "negative" in emo or any(k in emo for k in ("sad", "depressed", "hopeless")):
            state = "negative"
        else:
            state = "neutral"

        # Trigger detection for context-aware follow-ups.
        trigger_sleep = any(k in text_norm for k in ("sleep", "insomnia", "awake", "tired", "exhausted"))
        trigger_exam = any(k in text_norm for k in ("exam", "exams", "test", "study", "assignment"))
        trigger_family = any(k in text_norm for k in ("family", "parents", "mother", "father", "home"))
        trigger_lonely = any(k in text_norm for k in ("lonely", "loneliness", "alone", "isolated"))
        trigger_overthinking = any(k in text_norm for k in ("overthinking", "overthink", "thinking too much", "can't stop thinking"))
        trigger_future = any(k in text_norm for k in ("future", "career", "what if", "uncertain", "tomorrow"))
        trigger_work = any(k in text_norm for k in ("work", "job", "deadline", "boss", "office"))
        trigger_relationship = any(k in text_norm for k in ("friend", "partner", "relationship", "argument"))

        action_step = "Try one quick reset now: drink water, take 6 slow breaths, then reassess."
        follow_up = "What would make the next hour feel 10% easier?"
        if trigger_sleep:
            action_step = "Tonight, try a 30-minute wind-down: low light, no phone, and slow breathing."
            follow_up = "How has your sleep changed this week: trouble falling asleep, staying asleep, or waking too early?"
        elif trigger_exam:
            action_step = "Pick one 15-minute study block on a single topic, then take a 3-minute break."
            follow_up = "Which exam topic feels hardest right now, and what is one tiny step you can finish today?"
        elif trigger_family:
            action_step = "Write one calm sentence about what you need, so you can communicate it clearly."
            follow_up = "What part of the family situation is affecting you the most right now?"
        elif trigger_lonely:
            action_step = "Send one simple message to someone you trust, even: 'Can we talk later today?'"
            follow_up = "When does loneliness feel strongest for you—morning, evening, or night?"
        elif trigger_overthinking:
            action_step = "Try a 5-minute thought dump, then circle one thought you can act on right now."
            follow_up = "What thought keeps looping the most?"
        elif trigger_future:
            action_step = "List two things you can control today and one worry you will postpone until tomorrow."
            follow_up = "What future worry feels heaviest, and what is one step you can take this week?"
        elif trigger_relationship:
            action_step = "Write one gentle boundary or support request you can send today."
            follow_up = "What interaction triggered this feeling most recently?"
        elif trigger_work:
            action_step = "Pick one 10-minute task only, finish it, then pause before choosing the next step."
            follow_up = "Is the pressure mainly workload, deadlines, or expectations?"

        templates: dict[str, list[str]] = {
            "positive": [
                "I’m glad there is some light here. What helped most today that you can repeat tomorrow?\n\n{action}\n\n{follow_up}",
                "That sounds like a meaningful shift. Which part of your day made the biggest difference?\n\n{action}\n\n{follow_up}",
            ],
            "negative": [
                "That sounds heavy, and I’m glad you said it out loud. What feels hardest right now: thoughts, body, or situation?\n\n{action}\n\n{follow_up}",
                "I hear you. When this mood rises, what usually triggers it first?\n\n{action}\n\n{follow_up}",
            ],
            "anxious": [
                "That sounds very tense. Is your anxiety showing up more in your thoughts, your chest, or your breathing?\n\nTry this now: inhale 4s, exhale 6s, for 6 rounds.\n\n{follow_up}",
                "I can hear the pressure in this. What is one worry you can postpone for 30 minutes while you focus on one concrete task?\n\n{action}\n\n{follow_up}",
            ],
            "stress": [
                "You’re carrying a lot. What is one thing you can drop, delay, or simplify today?\n\n{action}\n\n{follow_up}",
                "That sounds draining. If we break this down, what is the smallest next step that still moves you forward?\n\n{action}\n\n{follow_up}",
            ],
            "depression": [
                "Thank you for sharing this. You don’t have to solve everything right now. What is one thing that would make this hour 10% easier?\n\n{action}\n\n{follow_up}\n\nIf you feel unsafe, contact local emergency or crisis support now.",
                "I’m really glad you reached out. Are you feeling more numb, sad, or exhausted right now?\n\n{action}\n\n{follow_up}\n\nIf you feel at risk of self-harm, please seek immediate crisis support.",
            ],
            "neutral": [
                "Thanks for sharing. What feeling is strongest right now, even if it is mild?\n\n{action}\n\n{follow_up}",
                "I’m here with you. What happened in the last hour that shifted your mood the most?\n\n{action}\n\n{follow_up}",
            ],
        }

        chosen = random.choice(templates.get(state, templates["neutral"]))

        if text:
            snippet = text.replace("\n", " ")
            if len(snippet) > 90:
                snippet = snippet[:87] + "..."
            prefix = f"You said: \"{snippet}\".\n\n"
        else:
            prefix = "Thanks for checking in.\n\n"

        reply = prefix + chosen.format(action=action_step, follow_up=follow_up)
        reply = reply.strip()
        if not reply:
            return fallback
        return reply
    except Exception:
        return fallback


def _api_chatbot_response(user_text: str, emotion_label: str, mental_pred: int) -> str:
    """
    Optional API response path using OpenAI-compatible Chat Completions endpoint.
    Used only when OPENAI_API_KEY is available.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return ""

    fallback = _local_rule_based_response(user_text, emotion_label, mental_pred)
    try:
        system_prompt = (
            "You are a supportive, empathetic, non-clinical mental health companion. "
            "Keep responses short (3-5 sentences), human, and practical. "
            "Use the provided emotion label and depression flag as context. "
            "Ask one thoughtful follow-up question and suggest one small action step. "
            "Do not be generic. Never leave the response empty."
        )
        context_prompt = (
            f"Detected emotion: {emotion_label}\n"
            f"Depression prediction flag: {int(mental_pred)}\n"
            f"User message: {user_text or '(no message provided)'}"
        )

        payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 220,
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return text if text else fallback
    except Exception:
        return fallback


def generate_chatbot_response(user_text: str, emotion_label: str, mental_pred: int) -> str:
    """
    Hybrid chatbot:
    - If API key is available, try API-generated response.
    - Otherwise, (or on failure) use local rule-based response.
    Always returns a non-empty string.
    """
    try:
        api_reply = _api_chatbot_response(user_text, emotion_label, mental_pred)
        if api_reply and api_reply.strip():
            return api_reply.strip()
        local_reply = _local_rule_based_response(user_text, emotion_label, mental_pred)
        return local_reply if local_reply.strip() else "I’m here with you. What feels most important to talk about right now?"
    except Exception:
        return "I’m here with you. What feels most important to talk about right now?"


def _chatbot_reply(emotion_label: str, mental_pred: int, user_text: str | None = None) -> str:
    """
    Compatibility wrapper used by existing dashboard flow.
    """
    return generate_chatbot_response(user_text or "", emotion_label, int(mental_pred))


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

    mental_health_vectorizer_path = model_dir / "mental_health_vectorizer.pkl"
    pre = TextPreprocessor()

    # Load models with clear error messages (avoid crashing on pickle/unpickling issues).
    try:
        emotion_model = _load_pickle(emotion_model_path)
        emotion_vectorizer = _load_pickle(emotion_vectorizer_path)
        mental_health_model = _load_pickle(mental_health_model_path)
    except Exception as e:
        st.error(f"Failed to load one or more models/vectorizers: {e}")
        st.info("If the files were modified recently, re-train and re-run the app.")
        return

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
        audio_file = st.file_uploader("Upload audio (wav/mp3, optional)", type=["wav", "mp3"])
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
                suffix = Path(audio_file.name).suffix.lower() if getattr(audio_file, "name", None) else ".wav"
                if suffix not in {".wav", ".mp3"}:
                    suffix = ".wav"
                tmp_path = model_dir / f"uploaded_audio_{int(datetime.now().timestamp())}{suffix}"
                tmp_path.write_bytes(audio_file.getbuffer())

                # SpeechRecognition's AudioFile is most reliable with WAV,
                # so if the user uploaded MP3 we convert it to a temporary WAV.
                transcription_path: Path = tmp_path
                tmp_wav_for_transcription: Path | None = None

                if suffix == ".mp3":
                    try:
                        tmp_wav_for_transcription = model_dir / f"transcription_{int(datetime.now().timestamp())}.wav"
                        audio_seg = AudioSegment.from_mp3(str(tmp_path))
                        audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
                        audio_seg.export(str(tmp_wav_for_transcription), format="wav")
                        transcription_path = tmp_wav_for_transcription
                    except Exception as e:
                        st.error(
                            "Could not convert MP3 for transcription. Ensure ffmpeg is installed and on your PATH."
                        )
                        return

                try:
                    st.info("Transcribing audio...")
                    # `speech_to_text()` raises ValueError when speech is unclear,
                    # and RuntimeError when the audio can't be processed.
                    transcribed_text = speech_to_text(str(transcription_path))
                    if not transcribed_text or not str(transcribed_text).strip():
                        st.warning("Transcription returned empty text.")
                        return

                    st.subheader("Converted Text (from audio)")
                    st.write(transcribed_text)
                    input_text = transcribed_text
                except ValueError as e:
                    st.warning(f"Could not understand the audio clearly: {e}")
                    return
                except RuntimeError as e:
                    st.error(f"Audio transcription failed: {e}")
                    return
                finally:
                    # Best-effort cleanup of temporary upload.
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    if tmp_wav_for_transcription is not None:
                        try:
                            tmp_wav_for_transcription.unlink(missing_ok=True)
                        except Exception:
                            pass
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
            emotion_label = _normalize_emotion_label(emotion_pred, processed_text=processed_text)

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

            st.subheader("Results")
            results_col1, results_col2 = st.columns(2, gap="large")

            with results_col1:
                st.markdown("#### Detected Emotion")
                if "positive" in emotion_label.lower():
                    st.success(emotion_label)
                elif "negative" in emotion_label.lower():
                    st.warning(emotion_label)
                else:
                    st.info(emotion_label)

            with results_col2:
                st.markdown("#### Mental Health Prediction")
                if mental_pred == 1:
                    st.error(mental_label)
                else:
                    st.success(mental_label)

            st.subheader("Recommendations")
            st.info(_get_recommendation(mental_pred=mental_pred, emotion_label=emotion_label))

            # Update session state
            st.session_state.last_emotion = emotion_label
            st.session_state.last_mental_pred = mental_pred
            st.session_state.mood_history.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "emotion": emotion_label,
                    "mental_pred": int(mental_pred),
                    "mental_label": "Depression" if mental_pred == 1 else "Not Depression",
                }
            )

            # Seed chatbot with an emotion-based assistant message
            assistant_msg = _chatbot_reply(emotion_label, mental_pred, user_text=None)
            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_msg})
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    with tabs[1]:
        st.subheader("History")
        history: list[dict[str, Any]] = st.session_state.mood_history
        if not history:
            st.caption("No history yet. Run an analysis first to build your trend.")
        else:
            df = pd.DataFrame(history)
            # Keep display tidy and predictable.
            display_cols = [c for c in ["timestamp", "emotion", "mental_label", "mental_pred"] if c in df.columns]
            chart_df = df["mental_pred"].tail(50).reset_index(drop=True) if "mental_pred" in df.columns else None
            depression_count = int(chart_df.sum()) if chart_df is not None else 0

            history_col1, history_col2 = st.columns([1.2, 1], gap="large")
            with history_col1:
                st.dataframe(df[display_cols].tail(50), use_container_width=True)
            with history_col2:
                st.metric("Depression flagged (last runs)", depression_count)
                if chart_df is not None:
                    st.caption("Trend (0 = not depression, 1 = depression) for the last runs.")
                    st.line_chart(chart_df)

            if st.button("Clear history", use_container_width=True):
                st.session_state.mood_history = []
                st.success("History cleared.")

    with tabs[2]:
        st.subheader("Chatbot")
        if not st.session_state.chat_messages:
            st.caption("Run an analysis first to enable emotion-based support messages.")

        st.divider()
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
