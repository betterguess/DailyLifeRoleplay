#!/usr/bin/env python3
"""Realtime microphone transcriber for the Streamlit app.

Exposes:
- ws://<host>:<port>/transcribe (JSON messages: {"partial": ...} / {"final": ...})
- http://<host>:<port>/final (latest final text)
- http://<host>:<port>/healthz
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from http import HTTPStatus

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.legacy.server import WebSocketServerProtocol, serve

LATEST_FINAL = ""
CLIENTS: set[WebSocketServerProtocol] = set()
STATE_LOCK = asyncio.Lock()


@dataclass
class RuntimeConfig:
    provider: str
    host: str
    port: int

    # Local provider settings
    sample_rate: int
    frame_ms: int
    vad_threshold: float
    silence_ms: int
    min_speech_ms: int
    partial_interval_ms: int
    model_size: str
    model_device: str
    model_compute_type: str
    language: str
    beam_size: int
    input_device: str

    # Azure provider settings
    azure_speech_key: str
    azure_speech_region: str
    azure_language: str
    list_input_devices: bool


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Realtime microphone transcriber service")
    parser.add_argument(
        "--provider",
        choices=["local", "azure", "auto"],
        default=os.getenv("STT_PROVIDER", "auto"),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000, help="Listen port (default: 9000)")

    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--frame-ms", type=int, default=30, help="Frame size for VAD")
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.015,
        help="RMS threshold for speech activity (normalized float)",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=700,
        help="Silence duration ending an utterance",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=450,
        help="Ignore speech chunks shorter than this",
    )
    parser.add_argument(
        "--partial-interval-ms",
        type=int,
        default=800,
        help="How often partial transcriptions are emitted while speaking",
    )
    parser.add_argument("--model-size", default="small", help="faster-whisper model size")
    parser.add_argument("--model-device", default="auto", help="Model device: auto/cpu/cuda")
    parser.add_argument(
        "--model-compute-type",
        default="int8",
        help="faster-whisper compute type (e.g. int8, float16)",
    )
    parser.add_argument("--language", default="da", help="Language code for local whisper (default: da)")
    parser.add_argument("--beam-size", type=int, default=1, help="Whisper beam size")
    parser.add_argument(
        "--input-device",
        default=os.getenv("STT_INPUT_DEVICE", ""),
        help="Local input device (index or name) for sounddevice",
    )
    parser.add_argument(
        "--list-input-devices",
        action="store_true",
        help="List available local audio input devices and exit",
    )

    parser.add_argument("--azure-speech-key", default=os.getenv("AZURE_SPEECH_KEY", ""))
    parser.add_argument("--azure-speech-region", default=os.getenv("AZURE_SPEECH_REGION", ""))
    parser.add_argument(
        "--azure-language",
        default=os.getenv("AZURE_SPEECH_LANGUAGE", "da-DK"),
        help="Azure STT language locale (default: da-DK)",
    )

    args = parser.parse_args()
    return RuntimeConfig(
        provider=args.provider,
        host=args.host,
        port=args.port,
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        vad_threshold=args.vad_threshold,
        silence_ms=args.silence_ms,
        min_speech_ms=args.min_speech_ms,
        partial_interval_ms=args.partial_interval_ms,
        model_size=args.model_size,
        model_device=args.model_device,
        model_compute_type=args.model_compute_type,
        language=args.language,
        beam_size=args.beam_size,
        input_device=args.input_device,
        azure_speech_key=args.azure_speech_key,
        azure_speech_region=args.azure_speech_region,
        azure_language=args.azure_language,
        list_input_devices=args.list_input_devices,
    )


async def broadcast(payload: dict[str, str]) -> None:
    if not CLIENTS:
        return

    message = json.dumps(payload, ensure_ascii=False)
    dead: list[WebSocketServerProtocol] = []
    for ws in CLIENTS:
        try:
            await ws.send(message)
        except ConnectionClosed:
            dead.append(ws)
        except Exception:
            dead.append(ws)

    for ws in dead:
        CLIENTS.discard(ws)


def _queue_event(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[tuple[str, str]],
    kind: str,
    text: str,
) -> None:
    text = text.strip()
    if not text:
        return

    def _push() -> None:
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        queue.put_nowait((kind, text))

    loop.call_soon_threadsafe(_push)


class LocalWhisperProvider:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # pragma: no cover - runtime dependency gate
            raise SystemExit(
                "Missing dependency: faster-whisper. Install with:\n"
                "  .venv/bin/pip install faster-whisper"
            ) from exc

        logging.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            cfg.model_size,
            cfg.model_device,
            cfg.model_compute_type,
        )
        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.model_device,
            compute_type=cfg.model_compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame))))


def capture_audio(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[np.ndarray],
    stop_event: asyncio.Event,
    cfg: RuntimeConfig,
) -> None:
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - runtime dependency gate
        raise SystemExit(
            "Missing dependency: sounddevice. Install with:\n"
            "  .venv/bin/pip install sounddevice"
        ) from exc

    frame_samples = int(cfg.sample_rate * (cfg.frame_ms / 1000))
    input_device: int | str | None = None
    if cfg.input_device.strip():
        raw = cfg.input_device.strip()
        input_device = int(raw) if raw.isdigit() else raw

    def _enqueue_frame(frame: np.ndarray) -> None:
        def _push() -> None:
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(frame)

        loop.call_soon_threadsafe(_push)

    def callback(indata, _frames, _time_info, status) -> None:
        if status:
            logging.debug("Audio status: %s", status)
        frame = np.asarray(indata[:, 0], dtype=np.float32).copy()
        _enqueue_frame(frame)

    with sd.InputStream(
        samplerate=cfg.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frame_samples,
        device=input_device,
        callback=callback,
    ):
        while not stop_event.is_set():
            time.sleep(0.05)


async def run_local_provider(cfg: RuntimeConfig, queue: asyncio.Queue[tuple[str, str]]) -> None:
    provider = LocalWhisperProvider(cfg)

    audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=256)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    capture_task = asyncio.create_task(
        asyncio.to_thread(capture_audio, loop, audio_queue, stop_event, cfg)
    )

    frame_ms = cfg.frame_ms
    silence_frames_limit = max(1, cfg.silence_ms // frame_ms)
    min_speech_frames = max(1, cfg.min_speech_ms // frame_ms)
    partial_interval_frames = max(1, cfg.partial_interval_ms // frame_ms)

    speaking = False
    frames: list[np.ndarray] = []
    silent_frames = 0
    since_partial = 0

    async def transcribe_emit(kind: str, wav: np.ndarray) -> None:
        text = (await asyncio.to_thread(provider.transcribe, wav)).strip()
        if text:
            await queue.put((kind, text))

    try:
        while True:
            frame = await audio_queue.get()
            voiced = rms(frame) >= cfg.vad_threshold

            if voiced:
                speaking = True
                silent_frames = 0
                frames.append(frame)
                since_partial += 1

                if len(frames) >= min_speech_frames and since_partial >= partial_interval_frames:
                    since_partial = 0
                    wav = np.concatenate(frames, axis=0)
                    await transcribe_emit("partial", wav)
                continue

            if not speaking:
                continue

            frames.append(frame)
            silent_frames += 1
            if silent_frames < silence_frames_limit:
                continue

            speaking = False
            if len(frames) >= min_speech_frames:
                wav = np.concatenate(frames, axis=0)
                await transcribe_emit("final", wav)
            frames.clear()
            silent_frames = 0
            since_partial = 0
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        capture_task.cancel()
        with contextlib.suppress(Exception):
            await capture_task


def run_azure_worker(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[tuple[str, str]],
    stop_event: asyncio.Event,
    cfg: RuntimeConfig,
) -> None:
    if not cfg.azure_speech_key or not cfg.azure_speech_region:
        raise RuntimeError(
            "Azure provider requires credentials. Set env vars or args:\n"
            "  AZURE_SPEECH_KEY, AZURE_SPEECH_REGION"
        )

    try:
        import azure.cognitiveservices.speech as speechsdk
    except Exception as exc:  # pragma: no cover - runtime dependency gate
        raise RuntimeError(
            "Missing dependency: azure-cognitiveservices-speech. Install with:\n"
            "  .venv/bin/pip install azure-cognitiveservices-speech"
        ) from exc

    speech_config = speechsdk.SpeechConfig(
        subscription=cfg.azure_speech_key,
        region=cfg.azure_speech_region,
    )
    speech_config.speech_recognition_language = cfg.azure_language

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    def on_recognizing(evt) -> None:
        text = getattr(evt.result, "text", "") or ""
        _queue_event(loop, queue, "partial", text)

    def on_recognized(evt) -> None:
        reason = getattr(evt.result, "reason", None)
        if reason == speechsdk.ResultReason.RecognizedSpeech:
            text = getattr(evt.result, "text", "") or ""
            _queue_event(loop, queue, "final", text)

    def on_canceled(evt) -> None:
        logging.warning("Azure STT canceled: %s", evt)

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)

    recognizer.start_continuous_recognition()
    logging.info("Azure STT started (language=%s)", cfg.azure_language)
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        recognizer.stop_continuous_recognition()


async def run_azure_provider(cfg: RuntimeConfig, queue: asyncio.Queue[tuple[str, str]]) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    worker_task = asyncio.create_task(
        asyncio.to_thread(run_azure_worker, loop, queue, stop_event, cfg)
    )
    try:
        await worker_task
    except asyncio.CancelledError:
        stop_event.set()
        worker_task.cancel()
        with contextlib.suppress(Exception):
            await worker_task


async def provider_loop(cfg: RuntimeConfig, queue: asyncio.Queue[tuple[str, str]]) -> None:
    if cfg.provider == "azure":
        await run_azure_provider(cfg, queue)
        return

    if cfg.provider == "local":
        await run_local_provider(cfg, queue)
        return

    # auto: prefer Azure when credentials are present; otherwise local fallback.
    if cfg.azure_speech_key and cfg.azure_speech_region:
        try:
            logging.info("Provider auto: trying Azure STT first")
            await run_azure_provider(cfg, queue)
            return
        except Exception as exc:
            logging.warning("Azure STT failed, falling back to local provider: %s", exc)
    else:
        logging.info("Provider auto: Azure credentials missing, using local provider")

    await run_local_provider(cfg, queue)


async def event_broadcast_loop(queue: asyncio.Queue[tuple[str, str]]) -> None:
    global LATEST_FINAL

    while True:
        kind, text = await queue.get()
        if kind == "final":
            async with STATE_LOCK:
                LATEST_FINAL = text
            await broadcast({"final": text})
            logging.info("Final: %s", text)
        else:
            await broadcast({"partial": text})


async def ws_handler(websocket: WebSocketServerProtocol, path: str) -> None:
    if path != "/transcribe":
        await websocket.close(code=1008, reason="Unknown path")
        return

    CLIENTS.add(websocket)
    logging.info("Client connected (%d total)", len(CLIENTS))
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)
        logging.info("Client disconnected (%d total)", len(CLIENTS))


async def process_request(path: str, _headers: object):
    if path == "/healthz":
        body = b"ok\n"
        headers = [
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ]
        return HTTPStatus.OK, headers, body

    if path == "/final":
        async with STATE_LOCK:
            payload = {"text": LATEST_FINAL}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Cache-Control", "no-store"),
            ("Content-Length", str(len(body))),
        ]
        return HTTPStatus.OK, headers, body

    return None


async def main() -> None:
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if cfg.list_input_devices:
        try:
            import sounddevice as sd
        except Exception as exc:
            raise SystemExit(
                "Missing dependency: sounddevice. Install with:\n"
                "  .venv/bin/pip install sounddevice"
            ) from exc

        devices = sd.query_devices()
        default_input = None
        with contextlib.suppress(Exception):
            default_input = sd.default.device[0]

        print("Available input devices:")
        for idx, dev in enumerate(devices):
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            marker = " (default)" if default_input == idx else ""
            name = dev.get("name", f"device-{idx}")
            chans = int(dev.get("max_input_channels", 0))
            sr = int(dev.get("default_samplerate", 0))
            print(f"  [{idx}] {name} | channels={chans} | default_sr={sr}{marker}")
        return

    logging.info(
        "Starting transcriber provider=%s on %s:%s (ws: /transcribe, http: /final) input_device=%s",
        cfg.provider,
        cfg.host,
        cfg.port,
        cfg.input_device.strip() or "default",
    )

    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=256)

    async with serve(ws_handler, cfg.host, cfg.port, process_request=process_request):
        await asyncio.gather(provider_loop(cfg, queue), event_broadcast_loop(queue))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
