from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionResult:
    label: str
    confidence: float


class EmotionDetector:
    """
    Placeholder emotion detector.
    Replace with a trained model (text, audio, or multimodal).
    """

    POSITIVE = {"happy", "joy", "excited", "grateful", "relieved"}
    NEGATIVE = {"sad", "depressed", "anxious", "angry", "hopeless", "tired", "panic"}

    def predict_from_text(self, text: str) -> EmotionResult:
        tokens = set(text.split())
        pos = len(tokens & self.POSITIVE)
        neg = len(tokens & self.NEGATIVE)
        if neg > pos:
            return EmotionResult(label="negative", confidence=min(0.55 + 0.1 * (neg - pos), 0.95))
        if pos > neg:
            return EmotionResult(label="positive", confidence=min(0.55 + 0.1 * (pos - neg), 0.95))
        return EmotionResult(label="neutral", confidence=0.55)
