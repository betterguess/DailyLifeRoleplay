import re
import socket
import threading
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
    """Build speech callback with Azure Speech first and local fallback."""
    try:
        import azure.cognitiveservices.speech as speechsdk

        def speak(text: str):
            clean_text = strip_emojis(text)
            speech_config = speechsdk.SpeechConfig(
                subscription=get_secret("AZURE_SPEECH_KEY"),
                region=get_secret("AZURE_SPEECH_REGION"),
            )
            speech_config.speech_synthesis_voice_name = "da-DK-JeppeNeural"
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )
            result = synthesizer.speak_text_async(clean_text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                st.audio(result.audio_data, format="audio/wav")
            else:
                st.error(f"TTS failed: {result.reason}")

        return speak
    except Exception:
        try:
            import pyttsx3

            engine = pyttsx3.init()

            def speak(text: str):
                engine.say(text)
                engine.runAndWait()

            return speak
        except ImportError:
            def speak(text: str):
                st.warning("TTS not available on this environment.")

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
