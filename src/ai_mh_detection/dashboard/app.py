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


# ── Condition → display name mapping ──────────────────────────────────────────
CONDITION_DISPLAY: dict[str, str] = {
    "normal":               "No Major Concern",
    "depression":           "Depression",
    "suicidal":             "Suicidal Ideation",
    "anxiety":              "Anxiety",
    "bipolar":              "Bipolar-related Signs",
    "stress":               "Stress",
    "personality disorder": "Personality-related Distress",
}

# Conditions that map to a "concern" (used for chatbot/history compatibility)
CONCERN_CONDITIONS: frozenset[str] = frozenset({"depression", "anxiety", "anger", "stress", "suicidal", "bipolar", "personality disorder"})


def _condition_from_pred(raw: object) -> str:
    """Normalise any model output to a lowercase condition string."""
    try:
        val = int(raw)
        # Legacy binary model: 1 = depression, 0 = normal
        return "depression" if val == 1 else "normal"
    except (TypeError, ValueError):
        pass
    return str(raw).strip().lower()


def _get_recommendation(*, condition: str, emotion_label: str) -> str:
    """
    Return a condition-aware, actionable recommendation.
    Works with multi-class labels (depression, anxiety, anger, happy, normal)
    and falls back gracefully for unknown labels.
    """
    cond = condition.strip().lower()
    emo = emotion_label.strip().lower()

    if cond == "depression":
        return (
            "**Depression detected — here is a 3-step support plan (10-20 min):**\n"
            "1. Reach out: message or call one trusted person — even just \"I'm having a rough day.\"\n"
            "2. Body reset: drink water, take a 5-minute walk or gentle stretch.\n"
            "3. Write 3 lines: what I feel / what I need / one tiny next step.\n\n"
            "If you feel unsafe or at risk of self-harm, please contact emergency or crisis support now."
        )

    if cond == "anxiety":
        return (
            "**Anxiety detected — quick relief steps (2-3 min):**\n"
            "1. Breathing: inhale 4 s, hold 1 s, exhale 6 s — repeat 6 times.\n"
            "2. Grounding: name 5 things you see, 4 you feel, 3 you hear.\n"
            "3. Narrow focus: choose one small, doable task for the next 10 minutes.\n\n"
            "If anxiety is persistent, speaking with a professional can help."
        )

    if cond == "anger":
        return (
            "**Anger / distress detected — a 3-step reset:**\n"
            "1. Pause: step away from the trigger for 5-10 minutes before reacting.\n"
            "2. Release: unclench jaw and fists, take 5 slow breaths, move your body briefly.\n"
            "3. Name it: what triggered this — is it the situation, or something deeper?\n\n"
            "If this pattern repeats, talking to someone supportive or a therapist can help."
        )

    if cond == "stress":
        return (
            "**Stress detected — clarity + recovery steps:**\n"
            "1. Identify the trigger: what happened right before this feeling?\n"
            "2. Release tension: unclench shoulders, relax hands, take 5 slow breaths.\n"
            "3. Simplify: pick ONE task to complete today — drop or delay everything else.\n\n"
            "If stress is ongoing, consider rest, delegation, or professional support."
        )

    if cond == "suicidal":
        return (
            "**This is serious — please reach out for support right now:**\n"
            "1. Contact a crisis line: International Association for Suicide Prevention "
            "lists resources at https://www.iasp.info/resources/Crisis_Centres/\n"
            "2. Tell someone you trust — a friend, family member, or mental health professional.\n"
            "3. Remove access to means if possible and stay in a safe environment.\n\n"
            "You are not alone. Professional help is available and effective."
        )

    if cond == "bipolar":
        return (
            "**Bipolar-related signs detected — steady routines help most:**\n"
            "1. Track your mood daily — a simple 1-10 scale reveals patterns.\n"
            "2. Protect sleep: irregular sleep is a major trigger for mood episodes.\n"
            "3. Speak with a psychiatrist if you are not already — medication management is often key.\n\n"
            "Bipolar disorder is very treatable with the right professional support."
        )

    if cond == "personality disorder":
        return (
            "**Personality-related distress detected — grounding and support matter:**\n"
            "1. Pause before reacting: take 3 slow breaths to create space between feeling and action.\n"
            "2. Name the emotion without acting on it immediately — feelings pass.\n"
            "3. Consider DBT-informed therapy if available — it is specifically effective here.\n\n"
            "These patterns are not character flaws; they are learnable skills."
        )

    if cond == "happy":
        return (
            "**Positive state detected — build on the momentum:**\n"
            "1. Note what helped — one specific thing you can repeat tomorrow.\n"
            "2. Share it: connection amplifies positive mood.\n"
            "3. Use this energy: tackle one meaningful task while you feel resourced.\n\n"
            "Keep monitoring — mood can shift. Regular check-ins help you stay ahead."
        )

    # normal / unknown
    if any(k in emo for k in ("positive", "happy", "joy")):
        return (
            "**No major concern — keep up the good work:**\n"
            "1. Maintain your routine: sleep, food, movement, connection.\n"
            "2. Note what's working and do more of it.\n"
            "3. Check in with yourself regularly — early awareness prevents bigger dips."
        )

    return (
        "**Things look steady — maintain your wellbeing:**\n"
        "1. Keep a basic routine: consistent sleep, meals, and movement.\n"
        "2. Do one low-effort check-in: \"How am I feeling in my body right now?\"\n"
        "3. If things shift downward, reach out early — support works best proactively."
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


def _local_rule_based_response(
    user_text: str,
    emotion_label: str,
    mental_pred: int,
    chat_history: list[dict[str, str]] | None = None,
    condition: str = "",
) -> str:
    """
    Local rule-based chatbot with rich, varied, context-aware responses.
    Uses conversation history to avoid repetition and improve continuity.
    Always returns a non-empty supportive response.
    """
    fallback = (
        "I'm here with you. What feels most present for you right now — "
        "a thought, a feeling, or a situation? Take your time."
    )

    try:
        text = (user_text or "").strip()
        text_norm = text.lower()
        emo = (emotion_label or "neutral").strip().lower()
        pred = int(mental_pred)
        history = chat_history or []

        # Collect recent assistant replies to avoid repeating the same phrasing.
        recent_bot_msgs = [
            m["content"] for m in history[-6:] if m.get("role") == "assistant"
        ]

        # Infer conversation state — condition string takes priority over binary pred.
        cond = condition.strip().lower()
        if cond == "suicidal":
            state = "suicidal"
        elif cond == "depression":
            state = "depression"
        elif cond == "bipolar":
            state = "bipolar"
        elif cond in ("anxiety",):
            state = "anxious"
        elif cond == "stress":
            state = "stress"
        elif cond == "personality disorder":
            state = "negative"
        elif pred == 1:
            state = "depression"  # legacy binary fallback
        elif "anxious" in emo or "panic" in emo or any(
            k in text_norm for k in ("anxious", "anxiety", "panic", "worried", "nervous", "dread")
        ):
            state = "anxious"
        elif "stress" in emo or any(
            k in text_norm for k in ("stressed", "overwhelmed", "deadline", "pressure", "burnout", "exhausted")
        ):
            state = "stress"
        elif "positive" in emo or any(k in emo for k in ("happy", "joy", "relieved", "excited", "grateful")):
            state = "positive"
        elif "negative" in emo or any(k in emo for k in ("sad", "depressed", "hopeless", "empty", "numb")):
            state = "negative"
        else:
            state = "neutral"

        # Topic triggers for specific action/follow-up selection.
        trigger_sleep = any(k in text_norm for k in ("sleep", "insomnia", "awake", "tired", "rest", "fatigue"))
        trigger_exam = any(k in text_norm for k in ("exam", "test", "study", "assignment", "grade", "college"))
        trigger_family = any(k in text_norm for k in ("family", "parents", "mother", "father", "home", "sibling"))
        trigger_lonely = any(k in text_norm for k in ("lonely", "loneliness", "alone", "isolated", "no one"))
        trigger_overthinking = any(k in text_norm for k in ("overthinking", "overthink", "spiral", "racing thoughts"))
        trigger_future = any(k in text_norm for k in ("future", "career", "what if", "uncertain", "tomorrow", "path"))
        trigger_work = any(k in text_norm for k in ("work", "job", "deadline", "boss", "office", "project"))
        trigger_relationship = any(k in text_norm for k in ("friend", "partner", "relationship", "argument", "breakup"))
        trigger_body = any(k in text_norm for k in ("body", "chest", "breathing", "heart", "headache", "stomach"))

        # Default action + follow-up (randomised so they vary per turn).
        action_step = random.choice([
            "Take a slow breath right now — in for 4 counts, out for 6. Then reassess.",
            "Drink a glass of water and step away for two minutes before continuing.",
            "Write down one thing you're feeling in one sentence — just naming it helps.",
            "Do a quick body scan: unclench your jaw, drop your shoulders, relax your hands.",
        ])
        follow_up = random.choice([
            "What would make the next hour feel even slightly more manageable?",
            "Is there one small thing that might help right now, even 5%?",
            "What's the heaviest thing sitting with you at this moment?",
            "What do you need most right now — to vent, think it through, or just be heard?",
        ])

        if trigger_sleep:
            action_step = random.choice([
                "Try a 20-minute wind-down tonight: dim lights, no phone, and slow breathing.",
                "Set a gentle alarm 30 minutes before bed as a signal to start winding down.",
                "Write out one worry before bed — offloading it from your head onto paper really helps.",
            ])
            follow_up = random.choice([
                "Is it more trouble falling asleep, staying asleep, or waking too early?",
                "How many nights this week has sleep felt off for you?",
                "What's usually running through your mind when you can't sleep?",
            ])
        elif trigger_exam:
            action_step = random.choice([
                "Pick the single hardest topic and do just 15 focused minutes on it — then stop.",
                "Write out what you already know about the topic. It's usually more than you think.",
                "Set a 25-minute timer, work on one thing only, then take a 5-minute real break.",
            ])
            follow_up = random.choice([
                "Which subject or topic feels most out of control right now?",
                "Is the stress more about the content itself, the time pressure, or both?",
                "What would 'good enough' preparation actually look like for you?",
            ])
        elif trigger_family:
            action_step = random.choice([
                "Write one sentence about what you actually need from this situation.",
                "Give yourself permission to step away physically for 10 minutes.",
                "Try to name the specific feeling — is it hurt, frustration, or disappointment?",
            ])
            follow_up = random.choice([
                "What part of the family dynamic is weighing on you most right now?",
                "How long has this particular tension been building?",
                "Is there someone in the situation you feel even slightly understood by?",
            ])
        elif trigger_lonely:
            action_step = random.choice([
                "Send one low-stakes message to someone — even just 'hey, thinking of you.'",
                "Spend 10 minutes somewhere with other people around, even without interacting.",
                "Write a few lines: what kind of connection are you missing most right now?",
            ])
            follow_up = random.choice([
                "Is the loneliness more about missing specific people, or a general disconnection?",
                "When was the last time you felt genuinely connected to someone?",
                "Are there people around but it still feels lonely? That's worth exploring.",
            ])
        elif trigger_overthinking:
            action_step = random.choice([
                "Set a 5-minute timer and write every thought down — then close the notebook.",
                "Pick the single most important thought and ask: can I act on this today, yes or no?",
                "Name the spiral out loud, then redirect to one physical sensation right now.",
            ])
            follow_up = random.choice([
                "What thought keeps looping back the most?",
                "Is the overthinking focused on the past, the present, or the future?",
                "What would you say to a close friend who was stuck in this same loop?",
            ])
        elif trigger_future:
            action_step = random.choice([
                "List two things within your control this week and one thing you'll let go of for now.",
                "Ask yourself: what's the very next smallest step — not the whole path, just the next one.",
                "Write the worry in one sentence, then write one thing you can do about it today.",
            ])
            follow_up = random.choice([
                "What's the specific future scenario worrying you most right now?",
                "Is this more about fear of failure, uncertainty, or something else entirely?",
                "What would 'good enough' look like if perfect isn't the goal?",
            ])
        elif trigger_relationship:
            action_step = random.choice([
                "Write one honest sentence about what you need from this relationship right now.",
                "Give yourself space before responding — 24 hours can genuinely shift perspective.",
                "Ask yourself: is what I'm feeling more about this person, or a pattern I've seen before?",
            ])
            follow_up = random.choice([
                "What happened that triggered this feeling most recently?",
                "Are you more hurt, angry, or confused — or a mix of all three?",
                "What would resolution or relief actually look like in this situation?",
            ])
        elif trigger_work:
            action_step = random.choice([
                "Pick the single most important task and do only that for the next 15 minutes.",
                "Write out what's actually on your plate — sometimes it's less scary when it's written down.",
                "Block 10 minutes as a real break — step outside, don't check messages.",
            ])
            follow_up = random.choice([
                "Is the pressure mainly about the volume, a specific deadline, or someone's expectations?",
                "How long have you been running at this pace without a real break?",
                "Is there anything on your list you could hand off, delay, or simplify?",
            ])
        elif trigger_body:
            action_step = random.choice([
                "Take three slow breaths: in through the nose, out through the mouth, longer exhale.",
                "Relax your shoulders, unclench your jaw, and put both feet flat on the floor.",
                "Step away from the screen for 5 minutes and move your body gently.",
            ])
            follow_up = random.choice([
                "Where in your body are you carrying most of the tension right now?",
                "Has your body been feeling this way for a while, or did something shift today?",
                "Physical and emotional stress are tightly linked — what do you think your body is reacting to?",
            ])

        # Rich template banks per emotional state (5-7 variations, no rigid fixed structure).
        templates: dict[str, list[str]] = {
            "positive": [
                "That's genuinely good to hear. It's easy to rush past moments like this — what's actually behind the feeling?\n\n{action}\n\n{follow_up}",
                "I notice things feel lighter right now. What made the difference today, even if it was small?\n\n{follow_up}",
                "Really glad you're in a better headspace. These moments matter — what helped you get here?\n\n{action}\n\n{follow_up}",
                "There's something to hold onto here. What do you want to remember about how this feels?\n\n{follow_up}",
                "Positive moments are worth pausing on, not just pushing through. What's contributing to this that you can keep going?\n\n{action}\n\n{follow_up}",
                "It's good to check in when things feel okay too — not just in hard moments. What's been going well?\n\n{follow_up}",
                "That kind of shift is real. What helped create it — something you did, something external, or just time?\n\n{action}\n\n{follow_up}",
            ],
            "negative": [
                "That sounds genuinely heavy. I'm glad you're putting it into words — that takes something. What feels hardest right now?\n\n{action}\n\n{follow_up}",
                "I hear you. Low moods like this can make everything feel harder than it actually is. What's the main thing pulling you down today?\n\n{action}\n\n{follow_up}",
                "It makes sense that you're feeling this way. You don't have to fix everything right now — just being here is enough.\n\n{action}\n\n{follow_up}",
                "Sometimes naming the feeling is the first real step. What word fits best — sad, empty, tired, heavy, or something else?\n\n{follow_up}",
                "I'm here and I'm listening. What's sitting with you most heavily right now?\n\n{action}\n\n{follow_up}",
                "Low periods don't last forever, even when they feel like they do. What's one thing that has helped even a little in the past?\n\n{action}\n\n{follow_up}",
                "You came here and shared this — that matters more than it might feel like right now. Is this a specific event, or more of a general heaviness?\n\n{action}\n\n{follow_up}",
            ],
            "anxious": [
                "That level of tension is real and it makes sense your system is on edge. Let's slow this down a little.\n\nTry this right now: breathe in for 4 counts, hold for 1, out for 6. Twice.\n\n{follow_up}",
                "Anxiety has a way of narrowing everything down to the worst possibility. What's the actual fear underneath this?\n\n{action}\n\n{follow_up}",
                "When anxiety peaks, the body responds before the mind can catch up. Are you feeling it more in your thoughts, your chest, or your gut?\n\n{action}\n\n{follow_up}",
                "That kind of worry is exhausting to carry. What's the one specific thing you're most anxious about right now?\n\n{action}\n\n{follow_up}",
                "Let's try to separate what you can control from what you can't. What part of this situation is actually within your hands?\n\n{action}\n\n{follow_up}",
                "Anxiety often focuses on future scenarios. What's the worst-case thought looping in your head — and how likely is it really?\n\n{follow_up}",
                "You're not alone in this. Anxiety amplifies threats — what do you know to be true about this situation right now?\n\n{action}\n\n{follow_up}",
            ],
            "stress": [
                "That's a lot to carry at once. Stress stacks up quietly until something tips. What's weighing on you most?\n\n{action}\n\n{follow_up}",
                "When everything feels urgent, nothing gets done well. Let's figure out one thing — just one — you can actually tackle today.\n\n{action}\n\n{follow_up}",
                "I can hear how stretched you are. Is the pressure coming from outside, from yourself, or both?\n\n{action}\n\n{follow_up}",
                "Burnout creeps in when we keep pushing without pausing. When did you last have a real break — not a distraction, an actual pause?\n\n{action}\n\n{follow_up}",
                "Stress makes everything feel equally urgent. What genuinely matters most right now, if you strip away the noise?\n\n{action}\n\n{follow_up}",
                "It sounds like you're running on empty. What's the smallest thing that would give you even 10 minutes of relief?\n\n{action}\n\n{follow_up}",
                "You're dealing with a lot. Let's not try to fix it all — what's one thing you can set aside, even temporarily?\n\n{action}\n\n{follow_up}",
            ],
            "depression": [
                "Thank you for sharing this with me. You don't have to be okay right now, and you don't have to solve everything at once. I'm here.\n\n{action}\n\n{follow_up}\n\n*If you ever feel unsafe, please reach out to a crisis line or emergency services.*",
                "What you're feeling is real. Depression makes even small things feel impossibly heavy. What's one thing — tiny, manageable — that might help this hour?\n\n{action}\n\n{follow_up}\n\n*If thoughts of self-harm arise, please contact a professional or crisis support immediately.*",
                "I really hear you. You reaching out matters, even if it doesn't feel that way. Are you feeling more numb, more sad, or more exhausted right now?\n\n{action}\n\n{follow_up}",
                "Depression often tells us we're alone in this — that's the lie it tells. Is there one person you trust, even a little, you could reach out to today?\n\n{action}\n\n{follow_up}\n\n*Your wellbeing matters. Professional support can make a real difference.*",
                "You don't have to explain yourself fully right now. What's one very small thing that has felt neutral or okay recently — even something tiny?\n\n{action}\n\n{follow_up}",
                "Sometimes when we're in this place, the basics matter most: water, light, gentle movement. Have you eaten and had water today?\n\n{action}\n\n{follow_up}\n\n*If you're struggling significantly, please consider speaking with a mental health professional.*",
                "I'm glad you're here and talking. Depression isolates — talking back to it, even like this, is a real step. What do you need most right now?\n\n{action}\n\n{follow_up}",
            ],
            "suicidal": [
                "I'm really glad you're here and that you're talking. What you're feeling right now is serious, and you deserve real support.\n\n"
                "Please reach out to a crisis line — they're free, confidential, and available now: https://www.iasp.info/resources/Crisis_Centres/\n\n"
                "Is there someone physically nearby you can be with right now?",
                "You matter, and what you're going through matters. I want you to be safe.\n\n"
                "The most important thing right now is to not be alone. Can you call or text someone you trust?\n\n"
                "If you feel in immediate danger, please call emergency services or go to your nearest emergency room.",
                "Thank you for sharing this with me — that took courage. You're not alone in this.\n\n"
                "Crisis support is available 24/7: https://www.iasp.info/resources/Crisis_Centres/\n\n"
                "What's happening right now that brought you to this point? I'm here to listen.",
                "I hear you, and I'm not going anywhere. What you're feeling is real — and it can get better with the right support.\n\n"
                "Please don't face this alone. Is there a counselor, therapist, or doctor you could contact today?\n\n"
                "If this is an emergency, please call your local emergency number now.",
            ],
            "bipolar": [
                "It sounds like things are shifting for you right now. Bipolar episodes — whether high or low — can feel very disorienting.\n\n{action}\n\n{follow_up}",
                "Mood swings can be exhausting, especially when they feel out of your control. What phase does this feel like for you right now — more up, more down, or mixed?\n\n{action}\n\n{follow_up}",
                "When you're in a high phase, everything feels possible but risky. When you're low, everything feels impossible. Which feels closer to right now?\n\n{action}\n\n{follow_up}",
                "One of the most helpful things with bipolar disorder is tracking. What's your sleep been like the past few nights?\n\n{action}\n\n{follow_up}",
                "These experiences are real and they're not your fault. Are you currently connected with a psychiatrist or mental health professional?\n\n{action}\n\n{follow_up}",
            ],
            "neutral": [
                "Thanks for checking in — it doesn't have to be a crisis to be worth talking about. What's on your mind?\n\n{action}\n\n{follow_up}",
                "Sometimes 'okay' is hiding something underneath. How are you actually doing, not just on the surface?\n\n{follow_up}",
                "I'm here. What's the most prominent thing in your head right now, even if it feels small?\n\n{action}\n\n{follow_up}",
                "Checking in regularly is one of the healthiest things you can do. What's shifted for you today, even slightly?\n\n{follow_up}",
                "What's the feeling that's been most present for you lately — even if it's been in the background?\n\n{action}\n\n{follow_up}",
                "You don't have to be in crisis to deserve support. What's something you've been carrying that you haven't talked about?\n\n{action}\n\n{follow_up}",
                "Sometimes the most useful question is: what do I actually need right now — to vent, think something through, or just feel heard?\n\n{follow_up}",
            ],
        }

        pool = templates.get(state, templates["neutral"])

        # Filter out responses too similar to recent bot messages (reduce repetition).
        def _is_too_similar(candidate: str) -> bool:
            candidate_words = set(candidate.lower().split())
            for prev in recent_bot_msgs:
                prev_words = set(prev.lower().split())
                overlap = candidate_words & prev_words
                threshold = max(8, 0.45 * min(len(candidate_words), len(prev_words)))
                if len(overlap) > threshold:
                    return True
            return False

        fresh_pool = [t for t in pool if not _is_too_similar(t)]
        chosen = random.choice(fresh_pool if fresh_pool else pool)

        reply = chosen.format(action=action_step, follow_up=follow_up).strip()

        # For follow-on messages (not the opening seed), optionally echo back a snippet
        # of what the user said to make the response feel more conversational.
        if text and len(history) > 1:
            snippet = text.replace("\n", " ").strip()
            if len(snippet) > 75:
                snippet = snippet[:72] + "..."
            ack_openers = [
                f'"{snippet}" — ',
                f"When you say that — ",
                f"I hear you on that. ",
                "",  # no ack sometimes feels most natural
                "",
            ]
            ack = random.choice(ack_openers)
            if ack and reply:
                reply = ack + reply[0].lower() + reply[1:]

        return reply if reply else fallback
    except Exception:
        return fallback



def _api_chatbot_response(user_text: str, emotion_label: str, mental_pred: int, condition: str = "") -> str:
    """
    Optional API response path using OpenAI-compatible Chat Completions endpoint.
    Used only when OPENAI_API_KEY is available.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return ""

    fallback = _local_rule_based_response(user_text, emotion_label, mental_pred, condition=condition)
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
            f"Detected mental health condition: {condition or ('concern' if mental_pred == 1 else 'no major concern')}\n"
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


def generate_chatbot_response(
    user_text: str,
    emotion_label: str,
    mental_pred: int,
    chat_history: list[dict[str, str]] | None = None,
    condition: str = "",
) -> str:
    """
    Hybrid chatbot entry point.
    - Uses OpenAI API if OPENAI_API_KEY is set.
    - Falls back to local rule-based responses with full template variation.
    - Passes conversation history for context-aware, non-repetitive replies.
    Always returns a non-empty string.
    """
    history = chat_history or []
    try:
        api_reply = _api_chatbot_response(user_text, emotion_label, mental_pred, condition=condition)
        if api_reply and api_reply.strip():
            return api_reply.strip()
        local_reply = _local_rule_based_response(user_text, emotion_label, mental_pred, history, condition=condition)
        return local_reply if local_reply.strip() else "I'm here with you. What feels most important to talk about right now?"
    except Exception:
        return "I'm here with you. What feels most important to talk about right now?"



def _chatbot_reply(
    emotion_label: str,
    mental_pred: int,
    user_text: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    condition: str = "",
) -> str:
    """Compatibility wrapper used by the dashboard flow."""
    return generate_chatbot_response(user_text or "", emotion_label, int(mental_pred), chat_history, condition=condition)


def main() -> None:
    cfg = _load_config().data
    st.set_page_config(
        page_title=cfg.get("app", {}).get("title", "AI Mental Health Detection"),
        layout="centered",
    )
    st.title(cfg.get("app", {}).get("title", "AI Mental Health Detection"))
    st.caption("Text-based emotion and mental health screening demo")
    st.caption(
        "Disclaimer: this tool is for informational and supportive screening purposes only. "
        "It is not a medical diagnosis and does not replace professional mental health care."
    )

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
    if "last_condition" not in st.session_state:
        st.session_state.last_condition = None

    tabs = st.tabs(["Analyze", "Mood History", "Chatbot"])

    with tabs[0]:
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

            # Parse the model output: the new multi-class model returns a string
            # label directly. The legacy binary model returns 0/1 — handled by
            # _condition_from_pred() for backward compatibility.
            condition = _condition_from_pred(mental_pred_raw)
            mental_label = CONDITION_DISPLAY.get(condition, condition.replace("_", " ").title())

            # Confidence score + top-3 breakdown (when model exposes predict_proba).
            confidence: float | None = None
            top_3: list[tuple[str, float]] | None = None
            try:
                if isinstance(mental_health_model, Pipeline) and hasattr(
                    mental_health_model, "predict_proba"
                ):
                    proba = mental_health_model.predict_proba([processed_text])[0]
                    classes = list(mental_health_model.classes_)
                    if condition in classes:
                        confidence = float(proba[classes.index(condition)])
                    sorted_preds = sorted(zip(classes, proba), key=lambda x: -x[1])
                    top_3 = [
                        (CONDITION_DISPLAY.get(c, c.replace("_", " ").title()), float(p))
                        for c, p in sorted_preds[:3]
                    ]
            except Exception:
                confidence = None
                top_3 = None

            # Compatibility integer (1 = concern, 0 = no concern) used by
            # chatbot seeding and mood history that pre-date multi-class.
            mental_pred = 1 if condition in CONCERN_CONDITIONS else 0


            st.success("Analysis complete.")

            st.subheader("Results")
            results_col1, results_col2 = st.columns(2, gap="large")

            with results_col1:
                st.markdown("#### Detected Emotion")
                if "positive" in emotion_label.lower() or emotion_label.lower() == "happy":
                    st.success(emotion_label.capitalize())
                elif any(k in emotion_label.lower() for k in ("negative", "angry", "anxious", "sad")):
                    st.warning(emotion_label.capitalize())
                else:
                    st.info(emotion_label.capitalize())

            with results_col2:
                st.markdown("#### Detected Condition")
                concern_conditions_display = {"depression", "anxiety", "anger", "stress", "suicidal", "bipolar", "personality disorder"}
                if condition in concern_conditions_display:
                    st.error(f"{mental_label}")
                elif condition == "happy":
                    st.success(f"{mental_label}")
                else:
                    st.info(f"{mental_label}")
                if confidence is not None:
                    pct = int(round(confidence * 100))
                    st.progress(confidence, text=f"Model confidence: {pct}%")
                if top_3 is not None and len(top_3) > 1:
                    with st.expander("Top-3 condition probabilities"):
                        for lbl, prob in top_3:
                            st.progress(prob, text=f"{lbl}: {int(round(prob * 100))}%")

            if condition == "suicidal":
                st.error(
                    "**Immediate support is strongly recommended.**\n\n"
                    "Signals of suicidal ideation were detected. This tool is not a crisis service — "
                    "please contact a crisis helpline now.\n\n"
                    "International resources: https://www.iasp.info/resources/Crisis_Centres/"
                )

            st.subheader("Recommendations")
            st.info(_get_recommendation(condition=condition, emotion_label=emotion_label))

            # Update session state
            st.session_state.last_emotion = emotion_label
            st.session_state.last_mental_pred = mental_pred
            st.session_state.last_condition = condition
            st.session_state.mood_history.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "emotion": emotion_label,
                    "condition": mental_label,
                    "mental_pred": int(mental_pred),
                    "confidence": f"{int(round(confidence * 100))}%" if confidence is not None else "N/A",
                }
            )

            # Seed chatbot with an emotion-based assistant message
            assistant_msg = _chatbot_reply(emotion_label, mental_pred, user_text=None, condition=condition)
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
            display_cols = [c for c in ["timestamp", "emotion", "condition", "confidence", "mental_pred"] if c in df.columns]
            chart_df = df["mental_pred"].tail(50).reset_index(drop=True) if "mental_pred" in df.columns else None
            depression_count = int(chart_df.sum()) if chart_df is not None else 0

            history_col1, history_col2 = st.columns([1.2, 1], gap="large")
            with history_col1:
                st.dataframe(df[display_cols].tail(50), use_container_width=True)
            with history_col2:
                st.metric("Concerns flagged (last runs)", depression_count)
                if chart_df is not None:
                    st.caption("Concern trend (0 = no concern, 1 = concern flagged) for the last runs.")
                    st.line_chart(chart_df)

            if st.button("Clear history", use_container_width=True):
                st.session_state.mood_history = []
                st.success("History cleared.")

    with tabs[2]:
        # ── Context banner ──────────────────────────────────────────────────────
        last_emotion = st.session_state.last_emotion
        last_mental_pred = st.session_state.last_mental_pred
        last_condition = st.session_state.get("last_condition")
        has_context = last_emotion is not None and last_mental_pred is not None

        if has_context:
            emotion_display = str(last_emotion).capitalize()
            condition_display = CONDITION_DISPLAY.get(
                str(last_condition), str(last_condition).replace("_", " ").title()
            ) if last_condition else ("Concern detected" if last_mental_pred == 1 else "No major concern")
            ctx_col1, ctx_col2, ctx_col3 = st.columns([2, 2, 1])
            with ctx_col1:
                if "positive" in str(last_emotion).lower() or last_emotion == "happy":
                    st.success(f"Emotion: {emotion_display}")
                elif any(k in str(last_emotion).lower() for k in ("negative", "anxious", "sad", "anger")):
                    st.warning(f"Emotion: {emotion_display}")
                else:
                    st.info(f"Emotion: {emotion_display}")
            with ctx_col2:
                concern_set = {"depression", "anxiety", "anger", "stress", "suicidal", "bipolar", "personality disorder"}
                if last_condition in concern_set:
                    st.error(f"Condition: {condition_display}")
                elif last_condition == "happy":
                    st.success(f"Condition: {condition_display}")
                else:
                    st.info(f"Condition: {condition_display}")
            with ctx_col3:
                if st.button("Clear chat", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.rerun()
        else:
            st.info(
                "Run an **analysis** first (Analyze tab) so the chatbot can respond based on "
                "your detected emotion and mental health state. You can still chat below — "
                "it will respond supportively using a neutral baseline."
            )
            _, clear_col = st.columns([4, 1])
            with clear_col:
                if st.button("Clear chat", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.rerun()

        st.divider()

        # ── Chat history display ────────────────────────────────────────────────
        if not st.session_state.chat_messages:
            st.caption("No messages yet. Say something below to start the conversation.")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ── Chat input ──────────────────────────────────────────────────────────
        placeholder = (
            "Type a message... (responding based on your detected emotion)"
            if has_context
            else "Type a message to start talking..."
        )
        user_chat = st.chat_input(placeholder)
        if user_chat:
            st.session_state.chat_messages.append({"role": "user", "content": user_chat})
            with st.chat_message("user"):
                st.markdown(user_chat)

            emotion_for_reply = str(last_emotion) if has_context else "neutral"
            mental_for_reply = int(last_mental_pred) if has_context else 0

            reply = _chatbot_reply(
                emotion_for_reply,
                mental_for_reply,
                user_text=user_chat,
                chat_history=st.session_state.chat_messages,
                condition=str(last_condition) if last_condition else "",
            )

            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

if __name__ == "__main__":
    main()
