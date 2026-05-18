"""
voice/tts.py
Text-to-Speech with multiple provider support.

Priority:
  1. ElevenLabs — most natural, low latency
  2. OpenAI TTS — good quality, fast
  3. gTTS       — free fallback

Returns audio bytes that can be streamed or saved.
"""

import os
import io
from typing import Optional


def synthesize_speech(text: str,
                       voice: str = "alloy",
                       provider: str = "gtts") -> bytes:
    """
    Converts text to speech audio bytes.

    provider: 'gtts'
    voice:    language code e.g. 'en'
    Returns: WAV/MP3 bytes
    """
    if provider == "gtts":
        return _gtts_tts(text)
    else:
        raise ValueError("Invalid provider")



def _gtts_tts(text: str, lang: str = "en") -> bytes:
    """
    Google TTS fallback — free, reasonable quality.
    """
    from gtts import gTTS
    tts    = gTTS(text=text, lang=lang, slow=False)
    buf    = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def clean_text_for_voice(text: str) -> str:
    """
    Strips markdown and formatting before TTS.
    TTS reads asterisks and pound signs aloud — clean them first.
    """
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # bold
    text = re.sub(r'\*(.+?)\*',     r'\1', text)   # italic
    text = re.sub(r'#{1,6}\s',      '',    text)   # headers
    text = re.sub(r'\[.+?\]\(.+?\)', '', text)     # links
    text = re.sub(r'`+.+?`+',       '', text)      # code
    text = re.sub(r'\|.+?\|',       '', text)      # tables
    text = re.sub(r'\n{2,}',     '. ', text)       # paragraph breaks
    text = re.sub(r'\n',          ' ', text)       # line breaks
    text = re.sub(r'\s{2,}',     ' ', text)        # extra spaces
    return text.strip()