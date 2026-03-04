#!/usr/bin/env python3
"""Realtime transcriber service.

Endpoints:
- ws://<host>:<port>/ingest      : browser audio input (PCM16 LE, mono)
- ws://<host>:<port>/transcribe  : transcript output ({"partial":...}/{"final":...})
- http://<host>:<port>/final     : latest final transcript
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
INGEST_QUEUE: asyncio.Queue[bytes] | None = None


@dataclass
class RuntimeConfig:
    provider: str
    audio_source: str
    host: str
    port: int
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
    list_input_devices: bool
    azure_speech_key: str
    azure_speech_region: str
    azure_language: str


class LocalWhisperProvider:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(
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


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Realtime microphone transcriber service")
    parser.add_argument(
        "--provider",
        choices=["local", "azure", "auto"],
        default=os.getenv("STT_PROVIDER", "auto"),
    )
    parser.add_argument(
        "--audio-source",
        choices=["browser", "local-mic"],
        default=os.getenv("STT_AUDIO_SOURCE", "browser"),
        help="browser: audio via /ingest websocket, local-mic: capture from host microphone",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=30)
    parser.add_argument("--vad-threshold", type=float, default=0.015)
    parser.add_argument("--silence-ms", type=int, default=700)
    parser.add_argument("--min-speech-ms", type=int, default=450)
    parser.add_argument("--partial-interval-ms", type=int, default=800)

    parser.add_argument("--model-size", default="small")
    parser.add_argument("--model-device", default="auto")
    parser.add_argument("--model-compute-type", default="int8")
    parser.add_argument("--language", default="da")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument(
        "--input-device",
        default=os.getenv("STT_INPUT_DEVICE", ""),
        help="Local microphone device name or index (local-mic mode)",
    )
    parser.add_argument("--list-input-devices", action="store_true")

    parser.add_argument("--azure-speech-key", default=os.getenv("AZURE_SPEECH_KEY", ""))
    parser.add_argument("--azure-speech-region", default=os.getenv("AZURE_SPEECH_REGION", ""))
    parser.add_argument("--azure-language", default=os.getenv("AZURE_SPEECH_LANGUAGE", "da-DK"))

    args = parser.parse_args()
    return RuntimeConfig(
        provider=args.provider,
        audio_source=args.audio_source,
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
        list_input_devices=args.list_input_devices,
        azure_speech_key=args.azure_speech_key,
        azure_speech_region=args.azure_speech_region,
        azure_language=args.azure_language,
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


async def enqueue_ingest_bytes(chunk: bytes) -> None:
    if not chunk or INGEST_QUEUE is None:
        return
    if INGEST_QUEUE.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            INGEST_QUEUE.get_nowait()
    await INGEST_QUEUE.put(chunk)


def rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame))))


async def run_local_with_array_chunks(cfg: RuntimeConfig, array_queue: asyncio.Queue[np.ndarray], event_queue: asyncio.Queue[tuple[str, str]]) -> None:
    model = LocalWhisperProvider(cfg)

    speaking = False
    chunks: list[np.ndarray] = []
    speech_ms = 0.0
    silence_ms = 0.0
    since_partial_ms = 0.0

    async def maybe_emit(kind: str, audio: np.ndarray) -> None:
        text = (await asyncio.to_thread(model.transcribe, audio)).strip()
        if text:
            await event_queue.put((kind, text))

    while True:
        samples = await array_queue.get()
        if samples.size == 0:
            continue

        chunk_ms = (samples.shape[0] / cfg.sample_rate) * 1000.0
        voiced = rms(samples) >= cfg.vad_threshold

        if voiced:
            speaking = True
            silence_ms = 0.0
            chunks.append(samples)
            speech_ms += chunk_ms
            since_partial_ms += chunk_ms

            if speech_ms >= cfg.min_speech_ms and since_partial_ms >= cfg.partial_interval_ms:
                since_partial_ms = 0.0
                audio = np.concatenate(chunks, axis=0)
                await maybe_emit("partial", audio)
            continue

        if not speaking:
            continue

        chunks.append(samples)
        silence_ms += chunk_ms
        if silence_ms < cfg.silence_ms:
            continue

        speaking = False
        if speech_ms >= cfg.min_speech_ms:
            audio = np.concatenate(chunks, axis=0)
            await maybe_emit("final", audio)
        chunks.clear()
        speech_ms = 0.0
        silence_ms = 0.0
        since_partial_ms = 0.0


def capture_audio_local_mic(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[np.ndarray], stop_event: asyncio.Event, cfg: RuntimeConfig) -> None:
    try:
        import sounddevice as sd
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: sounddevice. Install with:\n"
            "  .venv/bin/pip install sounddevice"
        ) from exc

    frame_samples = int(cfg.sample_rate * (cfg.frame_ms / 1000.0))
    input_device: int | str | None = None
    if cfg.input_device.strip():
        raw = cfg.input_device.strip()
        input_device = int(raw) if raw.isdigit() else raw

    def enqueue(frame: np.ndarray) -> None:
        def _push() -> None:
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(frame)
        loop.call_soon_threadsafe(_push)

    def callback(indata, _frames, _time_info, status) -> None:
        if status:
            logging.debug("Audio status: %s", status)
        enqueue(np.asarray(indata[:, 0], dtype=np.float32).copy())

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


async def run_local_provider(cfg: RuntimeConfig, event_queue: asyncio.Queue[tuple[str, str]]) -> None:
    chunk_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=256)

    if cfg.audio_source == "local-mic":
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        cap_task = asyncio.create_task(asyncio.to_thread(capture_audio_local_mic, loop, chunk_queue, stop_event, cfg))
        try:
            await run_local_with_array_chunks(cfg, chunk_queue, event_queue)
        finally:
            stop_event.set()
            cap_task.cancel()
            with contextlib.suppress(Exception):
                await cap_task
        return

    # browser ingest path: convert PCM16 bytes to float32 chunks in a feeder task.
    async def feeder() -> None:
        while True:
            if INGEST_QUEUE is None:
                await asyncio.sleep(0.05)
                continue
            raw = await INGEST_QUEUE.get()
            int16 = np.frombuffer(raw, dtype=np.int16)
            if int16.size == 0:
                continue
            samples = (int16.astype(np.float32) / 32768.0).copy()
            await chunk_queue.put(samples)

    feeder_task = asyncio.create_task(feeder())
    try:
        await run_local_with_array_chunks(cfg, chunk_queue, event_queue)
    finally:
        feeder_task.cancel()
        with contextlib.suppress(Exception):
            await feeder_task


async def run_azure_local_mic(cfg: RuntimeConfig, event_queue: asyncio.Queue[tuple[str, str]]) -> None:
    if not cfg.azure_speech_key or not cfg.azure_speech_region:
        raise RuntimeError("Azure provider requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")

    try:
        import azure.cognitiveservices.speech as speechsdk
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: azure-cognitiveservices-speech. Install with:\n"
            "  .venv/bin/pip install azure-cognitiveservices-speech"
        ) from exc

    loop = asyncio.get_running_loop()
    speech_config = speechsdk.SpeechConfig(subscription=cfg.azure_speech_key, region=cfg.azure_speech_region)
    speech_config.speech_recognition_language = cfg.azure_language
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    def on_recognizing(evt) -> None:
        text = getattr(evt.result, "text", "") or ""
        if text:
            loop.call_soon_threadsafe(event_queue.put_nowait, ("partial", text))

    def on_recognized(evt) -> None:
        reason = getattr(evt.result, "reason", None)
        if reason == speechsdk.ResultReason.RecognizedSpeech:
            text = getattr(evt.result, "text", "") or ""
            if text:
                loop.call_soon_threadsafe(event_queue.put_nowait, ("final", text))

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.start_continuous_recognition()
    logging.info("Azure STT started (local mic, language=%s)", cfg.azure_language)
    try:
        while True:
            await asyncio.sleep(0.1)
    finally:
        recognizer.stop_continuous_recognition()


async def run_azure_browser_ingest(cfg: RuntimeConfig, event_queue: asyncio.Queue[tuple[str, str]]) -> None:
    if not cfg.azure_speech_key or not cfg.azure_speech_region:
        raise RuntimeError("Azure provider requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")

    try:
        import azure.cognitiveservices.speech as speechsdk
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: azure-cognitiveservices-speech. Install with:\n"
            "  .venv/bin/pip install azure-cognitiveservices-speech"
        ) from exc

    loop = asyncio.get_running_loop()
    speech_config = speechsdk.SpeechConfig(subscription=cfg.azure_speech_key, region=cfg.azure_speech_region)
    speech_config.speech_recognition_language = cfg.azure_language

    fmt = speechsdk.audio.AudioStreamFormat(samples_per_second=cfg.sample_rate, bits_per_sample=16, channels=1)
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    def on_recognizing(evt) -> None:
        text = getattr(evt.result, "text", "") or ""
        if text:
            loop.call_soon_threadsafe(event_queue.put_nowait, ("partial", text))

    def on_recognized(evt) -> None:
        reason = getattr(evt.result, "reason", None)
        if reason == speechsdk.ResultReason.RecognizedSpeech:
            text = getattr(evt.result, "text", "") or ""
            if text:
                loop.call_soon_threadsafe(event_queue.put_nowait, ("final", text))

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.start_continuous_recognition()
    logging.info("Azure STT started (browser ingest, language=%s)", cfg.azure_language)

    try:
        while True:
            if INGEST_QUEUE is None:
                await asyncio.sleep(0.05)
                continue
            chunk = await INGEST_QUEUE.get()
            if chunk:
                push_stream.write(chunk)
    finally:
        with contextlib.suppress(Exception):
            push_stream.close()
        recognizer.stop_continuous_recognition()


async def run_azure_provider(cfg: RuntimeConfig, event_queue: asyncio.Queue[tuple[str, str]]) -> None:
    if cfg.audio_source == "local-mic":
        await run_azure_local_mic(cfg, event_queue)
    else:
        await run_azure_browser_ingest(cfg, event_queue)


async def provider_loop(cfg: RuntimeConfig, event_queue: asyncio.Queue[tuple[str, str]]) -> None:
    if cfg.provider == "azure":
        await run_azure_provider(cfg, event_queue)
        return

    if cfg.provider == "local":
        await run_local_provider(cfg, event_queue)
        return

    if cfg.azure_speech_key and cfg.azure_speech_region:
        try:
            logging.info("Provider auto: trying Azure STT first")
            await run_azure_provider(cfg, event_queue)
            return
        except Exception as exc:
            logging.warning("Azure STT failed, falling back to local provider: %s", exc)

    await run_local_provider(cfg, event_queue)


async def event_broadcast_loop(event_queue: asyncio.Queue[tuple[str, str]]) -> None:
    global LATEST_FINAL
    while True:
        kind, text = await event_queue.get()
        if kind == "final":
            async with STATE_LOCK:
                LATEST_FINAL = text
            await broadcast({"final": text})
            logging.info("Final: %s", text)
        elif kind == "partial":
            await broadcast({"partial": text})


async def ws_handler(websocket: WebSocketServerProtocol, path: str) -> None:
    if path == "/transcribe":
        CLIENTS.add(websocket)
        logging.info("Transcribe client connected (%d total)", len(CLIENTS))
        try:
            await websocket.wait_closed()
        finally:
            CLIENTS.discard(websocket)
            logging.info("Transcribe client disconnected (%d total)", len(CLIENTS))
        return

    if path == "/ingest":
        logging.info("Ingest client connected")
        try:
            async for message in websocket:
                if isinstance(message, (bytes, bytearray)):
                    await enqueue_ingest_bytes(bytes(message))
                # optional JSON control messages are ignored for now
        finally:
            logging.info("Ingest client disconnected")
        return

    await websocket.close(code=1008, reason="Unknown path")


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


def print_input_devices() -> None:
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


async def main() -> None:
    global INGEST_QUEUE

    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if cfg.list_input_devices:
        print_input_devices()
        return

    INGEST_QUEUE = asyncio.Queue(maxsize=512)
    event_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=512)

    logging.info(
        "Starting transcriber provider=%s source=%s on %s:%s (ws: /ingest + /transcribe, http: /final)",
        cfg.provider,
        cfg.audio_source,
        cfg.host,
        cfg.port,
    )

    async with serve(ws_handler, cfg.host, cfg.port, process_request=process_request, max_size=2_000_000):
        await asyncio.gather(provider_loop(cfg, event_queue), event_broadcast_loop(event_queue))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
