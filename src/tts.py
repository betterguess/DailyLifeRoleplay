import re
import socket
import threading
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

import streamlit as st

from src.config import get_secret


def strip_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def build_speak() -> Callable[[str], None]:
    """Build speech callback with Azure Speech primary and optional local fallback."""
    voice_name = os.getenv("AZURE_TTS_VOICE", "da-DK-JeppeNeural").strip() or "da-DK-JeppeNeural"
    allow_local_fallback = (
        os.getenv("ALLOW_LOCAL_TTS_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    try:
        import azure.cognitiveservices.speech as speechsdk
        azure_tts_lock = threading.Lock()
        logging.info("TTS backend: azure voice=%s", voice_name)

        def speak(text: str):
            clean_text = strip_emojis(text)
            if not clean_text.strip():
                return
            speech_config = speechsdk.SpeechConfig(
                subscription=get_secret("AZURE_SPEECH_KEY"),
                region=get_secret("AZURE_SPEECH_REGION"),
            )
            speech_config.speech_synthesis_voice_name = voice_name
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            try:
                with azure_tts_lock:
                    synthesizer = speechsdk.SpeechSynthesizer(
                        speech_config=speech_config,
                        audio_config=audio_config,
                    )
                    result = synthesizer.speak_text_async(clean_text).get()
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    return
                logging.warning("TTS failed: %s", result.reason)
            except Exception as exc:
                logging.warning("TTS exception: %s", exc)

        return speak
    except Exception as exc:
        logging.warning("Azure TTS unavailable: %s", exc)
        if allow_local_fallback:
            try:
                import pyttsx3

                engine = pyttsx3.init()
                local_tts_lock = threading.Lock()
                logging.info("TTS backend: pyttsx3 (local fallback enabled)")

                def speak(text: str):
                    if not (text or "").strip():
                        return
                    with local_tts_lock:
                        engine.say(text)
                        engine.runAndWait()

                return speak
            except ImportError:
                pass

        def speak(text: str):
            logging.warning("TTS unavailable (azure failed, local fallback disabled).")

        return speak


def _make_handler(
    speak: Callable[[str], None],
    expected_path: str = "/_tts",
):
    class _TTSHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                return
            query = parse_qs(parsed.query)
            text = query.get("text", [""])[0]
            if text:
                threading.Thread(target=speak, args=(text,), daemon=True).start()
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return _TTSHandler


def start_tts_server(
    speak: Callable[[str], None],
    *,
    host: str = "127.0.0.1",
    start_port: int = 8502,
    end_port: int = 8510,
) -> Optional[int]:
    for port in range(start_port, end_port + 1):
        try:
            sock = socket.socket()
            sock.bind((host, port))
            sock.close()

            handler = _make_handler(speak)

            def _run():
                try:
                    server = HTTPServer((host, port), handler)
                    server.serve_forever()
                except Exception:
                    pass

            threading.Thread(target=_run, daemon=True).start()
            return port
        except OSError:
            continue
    return None


def ensure_hover_tts_server(
    speak: Callable[[str], None],
    *,
    port: int,
    host: str = "127.0.0.1",
) -> None:
    handler = _make_handler(speak)

    def _run():
        try:
            httpd = HTTPServer((host, port), handler)
            httpd.serve_forever()
        except OSError:
            pass

    threading.Thread(target=_run, daemon=True).start()
