from __future__ import annotations

from dataclasses import dataclass

from ai_mh_detection.prediction import PredictionResult


@dataclass(frozen=True)
class ChatbotResponse:
    message: str
    safety_level: str  # "normal" | "elevated"


class SupportChatbot:
    """
    Simple, non-clinical supportive chatbot scaffold.
    If you add an LLM, keep a safety layer between user input and generation.
    """

    def respond(self, user_text: str, prediction: PredictionResult | None = None) -> ChatbotResponse:
        if prediction is not None and prediction.label == "high_risk":
            return ChatbotResponse(
                message=(
                    "I’m really sorry you’re feeling this way. If you might be in danger or "
                    "considering self-harm, please contact your local emergency number right now "
                    "or reach out to someone you trust. If you share your country, I can help "
                    "find crisis resources. If you’re safe in this moment, what’s been going on?"
                ),
                safety_level="elevated",
            )

        return ChatbotResponse(
            message=(
                "Thanks for sharing. I’m here with you. "
                "What’s the main thing you’ve been dealing with lately?"
            ),
            safety_level="normal",
        )
