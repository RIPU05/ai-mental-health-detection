from __future__ import annotations

from dataclasses import dataclass

from ai_mh_detection.prediction import PredictionResult


@dataclass(frozen=True)
class Recommendation:
    title: str
    description: str


class Recommender:
    def recommend(self, prediction: PredictionResult) -> list[Recommendation]:
        if prediction.label == "high_risk":
            return [
                Recommendation(
                    title="Reach out for urgent help",
                    description=(
                        "If you feel unsafe, contact local emergency services or a crisis line. "
                        "Consider reaching out to a trusted person nearby."
                    ),
                ),
                Recommendation(
                    title="Grounding (2 minutes)",
                    description="Name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
                ),
            ]
        if prediction.label == "moderate_risk":
            return [
                Recommendation(
                    title="Short breathing exercise",
                    description="Inhale 4s, hold 4s, exhale 6s. Repeat for 3–5 minutes.",
                ),
                Recommendation(
                    title="Journaling prompt",
                    description="What’s been hardest this week, and what’s one small step that could help?",
                ),
            ]
        return [
            Recommendation(
                title="Maintain routines",
                description="Sleep, hydration, and a short walk can help regulate mood.",
            ),
            Recommendation(
                title="Check-in",
                description="If things worsen, consider talking to a professional or trusted person.",
            ),
        ]
