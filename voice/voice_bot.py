"""
voice/voice_bot.py
Voice bot orchestrator.
Combines STT → ChatEngine → TTS into one pipeline.

Handles:
  - Real-time microphone input
  - Interruption detection
  - Low-latency response streaming
  - Session persistence
"""

import time
from chatbot.chat_agent import chatengine
from voice.stt import transcribe_audio_file, record_from_microphone
from voice.tts import synthesize_speech, clean_text_for_voice

class VoiceBot:
    """
    Stateful voice bot with session_id persistence.

    Usage (microphone mode):
      bot = VoiceBot(vectorstore)
      session_id = bot.new_session()
      bot.run_realtime_loop(session_id)

    Usage (API mode — audio bytes in, audio bytes out):
      bot = VoiceBot(vectorstore)
      session_id = bot.new_session()
      audio_out = bot.process_audio(session_id, audio_bytes)
    """

    def __init__(self, vectorstore,
                 tts_voice: str = "alloy",
                 tts_provider: str = "gtts"):
        self.engine       = chatengine(vectorstore, voice_mode=True)
        self.tts_voice    = tts_voice
        self.tts_provider = tts_provider

    def new_session(self) -> str:
        """Creates new voice session. Returns session_id."""
        return self.engine.new_session(mode="voice")

    def process_audio(self, session_id: str,
                      audio_bytes: bytes,
                      language: str = "en") -> dict:
        """
        Full pipeline for API voice mode:
          1. STT — transcribe audio bytes → text
          2. Chat — generate response text
          3. TTS — synthesize response → audio bytes

        Returns dict with transcript, response text, and audio bytes.
        """
        # Step 1 — STT
        start_stt = time.time()
        transcript = transcribe_audio_file(audio_bytes, language=language)
        stt_ms     = round((time.time() - start_stt) * 1000)

        if not transcript.strip():
            return {
                "session_id": session_id,
                "transcript": "",
                "response":   "",
                "audio":      b"",
                "error":      "No speech detected",
            }

        print(f"[VoiceBot] STT ({stt_ms}ms): {transcript}")

        # Step 2 — Chat
        start_llm = time.time()
        result    = self.engine.chat(session_id, transcript)
        llm_ms    = round((time.time() - start_llm) * 1000)
        response_text = result["response"]

        print(f"[VoiceBot] LLM ({llm_ms}ms): {response_text[:80]}...")

        # Step 3 — TTS
        clean_response = clean_text_for_voice(response_text)
        start_tts      = time.time()
        audio_out      = synthesize_speech(
            clean_response,
            voice    = self.tts_voice,
            provider = self.tts_provider,
        )
        tts_ms = round((time.time() - start_tts) * 1000)

        print(f"[VoiceBot] TTS ({tts_ms}ms): {len(audio_out)} bytes")

        return {
            "session_id": session_id,
            "transcript": transcript,
            "response":   response_text,
            "audio":      audio_out,
            "latency": {
                "stt_ms": stt_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
                "total_ms": stt_ms + llm_ms + tts_ms,
            },
            "sources": result.get("sources", []),
        }

    def run_realtime_loop(self, session_id: str):
        """
        Real-time microphone loop for desktop/terminal mode.
        Press Ctrl+C to exit.
        """
        print(f"\n[VoiceBot] Session: {session_id}")
        print("[VoiceBot] Speak now. Ctrl+C to stop.\n")

        while True:
            try:
                # Record from microphone
                transcript = record_from_microphone(
                    duration_seconds  = 10,
                    silence_threshold = 500,
                )

                if not transcript.strip():
                    print("[VoiceBot] No speech detected. Listening again...")
                    continue

                print(f"You: {transcript}")

                # Generate response
                result        = self.engine.chat(session_id, transcript)
                response_text = result["response"]
                print(f"BlackBot: {response_text}\n")

                # Speak response
                clean_response = clean_text_for_voice(response_text)
                audio_bytes    = synthesize_speech(
                    clean_response,
                    voice    = self.tts_voice,
                    provider = self.tts_provider,
                )
                self._play_audio(audio_bytes)

            except KeyboardInterrupt:
                print("\n[VoiceBot] Session ended.")
                break

    def _play_audio(self, audio_bytes: bytes):
        """Plays audio bytes through system speakers."""
        try:
            import pygame
            import io
            pygame.mixer.init()
            sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
            sound.play()
            while pygame.mixer.get_busy():
                pygame.time.wait(100)
        except ImportError:
            # fallback — save to temp file and play
            import tempfile, os, subprocess
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            subprocess.run(["start", tmp_path], shell=True)

    