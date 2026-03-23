from __future__ import annotations

from dataclasses import dataclass

from ai_mh_detection.emotion import EmotionResult


@dataclass(frozen=True)
class PredictionResult:
    label: str
    score: float


class MentalHealthPredictor:
    """
    Placeholder mental health predictor.
    Replace with a calibrated classifier/regressor trained on your dataset.
    """

    def predict(self, processed_text: str, emotion: EmotionResult | None = None) -> PredictionResult:
        # Very small heuristic baseline for scaffold purposes only.
        risk = 0.2
        high_risk_markers = [
            "suicide",
            "kill myself",
            "self harm",
            "worthless",
            "can't go on",
            "hopeless",
        ]
        t = processed_text
        if any(m in t for m in high_risk_markers):
            risk = 0.95
        elif emotion is not None and emotion.label == "negative":
            risk = min(0.75, 0.4 + 0.6 * emotion.confidence)
        elif emotion is not None and emotion.label == "positive":
            risk = 0.15

        if risk >= 0.75:
            return PredictionResult(label="high_risk", score=risk)
        if risk >= 0.4:
            return PredictionResult(label="moderate_risk", score=risk)
        return PredictionResult(label="low_risk", score=risk)
