from __future__ import annotations

import speech_recognition as sr


def speech_to_text(audio_path: str, *, language: str = "en-US") -> str:
    """
    Convert an audio file to text using the SpeechRecognition library.

    Args:
        audio_path: Path to an audio file (wav/mp3/etc).
        language: BCP-47 language code for the recognizer (default: "en-US").

    Returns:
        Transcribed text.

    Raises:
        FileNotFoundError: If the audio path does not exist.
        ValueError: If speech is detected but cannot be understood.
        RuntimeError: If the audio cannot be processed or the recognizer fails.
    """
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_path) as source:
            # Required by the spec: use recognizer.record(source)
            audio_data = recognizer.record(source)
    except FileNotFoundError:
        raise
    except sr.AudioFileError as e:
        raise RuntimeError(f"Could not process audio file: {audio_path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load/record audio: {audio_path}") from e

    try:
        # Default to online Google Web Speech recognition.
        return recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError as e:
        raise ValueError("Speech could not be understood from the audio.") from e
    except sr.RequestError as e:
        raise RuntimeError("Speech recognition request failed (check internet/API settings).") from e


__all__ = ["speech_to_text"]

