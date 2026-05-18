"""
voice/stt.py
Speech-to-Text using OpenAI Whisper.
Falls back to SpeechRecognition + Google STT if Whisper unavailable.

Supports:
  - File-based transcription (uploaded audio)
  - Real-time microphone capture
  - Interruption detection (silence threshold)
"""
import os
import io
import wave
from typing import Optional

def transcribe_audio_file(audio_bytes:bytes, language: str = 'en')->str:
    return _google_stt_transcribe(audio_bytes)


def _google_stt_transcribe(audio_bytes:bytes) -> str:
    import speech_recognition as sr
    recognizer = sr.recognizer()
    audio_file = io.BytesIO(audio_bytes)

    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return "Error: Could not request results from Google STT service"
    except Exception as e:
        return f"Error: {str(e)}"

def record_from_microphone(duration_seconds: int = 5,
                            silence_threshold: float = 500,
                            language: str = "en") -> str:
    """
    Records from microphone and transcribes.
    Stops early if silence detected (interruption handling).

    Used in local/desktop voice bot mode.
    """
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = silence_threshold
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8   # stop after 0.8s silence

    with sr.Microphone() as source:
        print("[STT] Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(
                source,
                timeout=duration_seconds,
                phrase_time_limit=30,
            )
            print("[STT] Processing...")
        except sr.WaitTimeoutError:
            return ""

    
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return ""