from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextPreprocessor:
    lowercase: bool = True
    remove_punctuation: bool = True
    collapse_whitespace: bool = True

    def transform(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s']", " ", text)
        if self.collapse_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text
