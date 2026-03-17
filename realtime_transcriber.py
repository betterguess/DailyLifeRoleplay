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
import re
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
LAST_INGEST_AT = 0.0
AZURE_STT_RESET_EVENT: asyncio.Event | None = None


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
    azure_segmentation_silence_ms: int
    log_file: str
    azure_allow_fallback_final: bool


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
    parser.add_argument(
        "--azure-segmentation-silence-ms",
        type=int,
        default=int(os.getenv("AZURE_SEGMENTATION_SILENCE_MS", "1600")),
        help="Pause duration before Azure emits final segment (ms)",
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("STT_LOG_FILE", ""),
        help="Optional file path for transcriber logs",
    )
    parser.add_argument(
        "--azure-allow-fallback-final",
        action="store_true",
        help="Allow partial->final fallback while using Azure provider (default: disabled)",
    )

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
        azure_segmentation_silence_ms=args.azure_segmentation_silence_ms,
        log_file=args.log_file,
        azure_allow_fallback_final=args.azure_allow_fallback_final,
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
    global LAST_INGEST_AT
    if not chunk or INGEST_QUEUE is None:
        return
    LAST_INGEST_AT = time.time()
    if INGEST_QUEUE.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            INGEST_QUEUE.get_nowait()
    await INGEST_QUEUE.put(chunk)


async def flush_ingest_queue() -> int:
    if INGEST_QUEUE is None:
        return 0
    dropped = 0
    while True:
        try:
            INGEST_QUEUE.get_nowait()
            dropped += 1
        except asyncio.QueueEmpty:
            break
    return dropped


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
    with contextlib.suppress(Exception):
        speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            str(cfg.azure_segmentation_silence_ms),
        )
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
    with contextlib.suppress(Exception):
        speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            str(cfg.azure_segmentation_silence_ms),
        )

    fmt = speechsdk.audio.AudioStreamFormat(samples_per_second=cfg.sample_rate, bits_per_sample=16, channels=1)

    def start_recognizer():
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
        return push_stream, recognizer

    def stop_recognizer(push_stream, recognizer) -> None:
        with contextlib.suppress(Exception):
            push_stream.close()
        with contextlib.suppress(Exception):
            recognizer.stop_continuous_recognition()

    push_stream, recognizer = start_recognizer()
    logging.info("Azure STT started (browser ingest, language=%s)", cfg.azure_language)

    try:
        while True:
            if AZURE_STT_RESET_EVENT is not None and AZURE_STT_RESET_EVENT.is_set():
                AZURE_STT_RESET_EVENT.clear()
                stop_recognizer(push_stream, recognizer)
                push_stream, recognizer = start_recognizer()
                logging.info("Azure STT recognizer reset after final")

            if INGEST_QUEUE is None:
                await asyncio.sleep(0.05)
                continue
            try:
                chunk = await asyncio.wait_for(INGEST_QUEUE.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            if chunk:
                await asyncio.to_thread(push_stream.write, chunk)
    finally:
        stop_recognizer(push_stream, recognizer)


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


async def event_broadcast_loop(cfg: RuntimeConfig, event_queue: asyncio.Queue[tuple[str, str]]) -> None:
    global LATEST_FINAL
    pending_partial = ""
    pending_partial_at = 0.0
    last_final_text = ""
    last_final_norm = ""
    last_final_source = ""
    last_final_at = 0.0
    final_history: list[str] = []
    partial_finalize_after_s = 2.2
    ingest_idle_before_finalize_s = 0.7
    duplicate_final_window_s = 2.0
    short_fallback_after_s = 6.0

    fallback_enabled = True
    if cfg.provider == "azure":
        # More conservative fallback settings for Azure browser ingest.
        partial_finalize_after_s = 3.8
        ingest_idle_before_finalize_s = 2.2
        # Keep fallback available in browser mode to prevent partial-only stalls.
        # The short-utterance guards below prevent early mid-sentence finals.
        if cfg.audio_source == "browser":
            fallback_enabled = True

    def _normalize_words(text: str) -> list[str]:
        # Tokenize to words to avoid punctuation/casing breaking prefix matches.
        return [w.lower() for w in re.findall(r"\w+", (text or "").strip(), flags=re.UNICODE)]

    def _normalize_text(text: str) -> str:
        return " ".join(_normalize_words(text))

    def _looks_incomplete_phrase(text: str) -> bool:
        words = _normalize_words(text)
        if not words:
            return False
        tail = words[-1]
        short_complete = {"ja", "ok", "nej", "tak"}
        if len(tail) <= 2 and tail not in short_complete:
            return True
        return tail in {
            "i",
            "og",
            "at",
            "til",
            "med",
            "på",
            "om",
            "for",
            "af",
            "en",
            "et",
            "den",
            "det",
            "de",
            "the",
        }

    def _is_noise_fragment(text: str) -> bool:
        words = _normalize_words(text)
        if not words:
            return True
        if len(words) == 1 and len(words[0]) <= 1:
            return True
        return False

    def _should_block_short_final(text: str) -> bool:
        words = _normalize_words(text)
        if not words:
            return True
        if _looks_incomplete_phrase(text):
            return True

        if len(words) == 1:
            allowed_single_word = {
                "ja",
                "nej",
                "tak",
                "kort",
                "langt",
                "spidserne",
                "maskine",
                "saks",
                "mellem",
                "stor",
                "lille",
                "kontant",
                "kortet",
            }
            return words[0] not in allowed_single_word

        # Avoid finals ending in a likely cut-off token.
        short_complete = {"ja", "ok", "nej", "tak"}
        if len(words[-1]) <= 2 and words[-1] not in short_complete:
            return True
        return False

    def _trim_repeated_prefix(text: str, prev_final: str) -> str:
        """
        Azure continuous recognition can emit text with previous final as prefix.
        Trim that prefix to keep each turn focused on newly spoken content.
        """
        text_words = _normalize_words(text)
        prev_words = _normalize_words(prev_final)
        if len(prev_words) < 2 or len(text_words) <= len(prev_words):
            return (text or "").strip()

        match = True
        for idx in range(len(prev_words)):
            if text_words[idx].lower() != prev_words[idx].lower():
                match = False
                break

        if not match:
            return (text or "").strip()

        trimmed = " ".join(text_words[len(prev_words):]).strip()
        return trimmed if trimmed else (text or "").strip()

    def _trim_known_prefix(text: str) -> str:
        """
        If Azure returns a partial/final that starts with a previously emitted
        utterance, keep only the newly appended suffix.
        """
        text_norm = _normalize_text(text)
        if len(text_norm.split()) < 2 or not final_history:
            return (text or "").strip()

        best_prefix = ""
        for prev in reversed(final_history[-10:]):
            prev_norm = _normalize_text(prev)
            if len(prev_norm.split()) < 2:
                continue
            if text_norm == prev_norm:
                continue
            if text_norm.startswith(prev_norm + " ") and len(prev_norm) > len(best_prefix):
                best_prefix = prev_norm

        if not best_prefix:
            return (text or "").strip()

        trimmed = text_norm[len(best_prefix):].strip()
        return trimmed if trimmed else (text or "").strip()

    async def emit_final(text: str, source: str) -> None:
        nonlocal last_final_text, last_final_norm, last_final_source, last_final_at, final_history
        text = (text or "").strip()
        if not text:
            return
        if cfg.provider == "azure" and cfg.audio_source == "browser":
            text = _trim_repeated_prefix(text, last_final_text)
            text = _trim_known_prefix(text)
            text = (text or "").strip()
            if not text:
                return
        now = time.time()
        text_norm = _normalize_text(text)
        if not text_norm:
            return

        if (text == last_final_text or text_norm == last_final_norm) and (now - last_final_at) < duplicate_final_window_s:
            return

        # Azure may emit fallback and provider finals for same utterance within
        # a short window. Suppress the second near-duplicate final.
        if (
            cfg.provider == "azure"
            and cfg.audio_source == "browser"
            and last_final_norm
            and (now - last_final_at) < 3.0
        ):
            similar = (
                text_norm.startswith(last_final_norm + " ")
                or last_final_norm.startswith(text_norm + " ")
            )
            if similar:
                logging.info(
                    "Final suppressed (near-duplicate %s->%s): %s",
                    last_final_source or "unknown",
                    source,
                    text,
                )
                return

        last_final_text = text
        last_final_norm = text_norm
        last_final_source = source
        last_final_at = now
        if cfg.provider == "azure" and cfg.audio_source == "browser":
            final_history.append(text)
            if len(final_history) > 20:
                del final_history[: len(final_history) - 20]
        async with STATE_LOCK:
            global LATEST_FINAL
            LATEST_FINAL = text
        await broadcast({"final": text})
        if cfg.provider == "azure" and cfg.audio_source == "browser":
            dropped = await flush_ingest_queue()
            if dropped > 0:
                logging.info("Ingest queue flushed after final (%s): dropped_chunks=%d", source, dropped)
            if AZURE_STT_RESET_EVENT is not None:
                AZURE_STT_RESET_EVENT.set()
        logging.info("Final (%s): %s", source, text)

    while True:
        try:
            kind, text = await asyncio.wait_for(event_queue.get(), timeout=0.25)
        except asyncio.TimeoutError:
            now = time.time()
            partial_stale = pending_partial and (now - pending_partial_at) >= partial_finalize_after_s
            ingest_idle = (now - LAST_INGEST_AT) >= ingest_idle_before_finalize_s
            if fallback_enabled and partial_stale and ingest_idle:
                text = pending_partial.strip()
                if _is_noise_fragment(text):
                    logging.info("Fallback suppressed (noise fragment): %s", text)
                    continue
                words = len(text.split())
                if cfg.provider == "azure" and cfg.audio_source == "browser":
                    # Allow short but valid Danish responses like "med maskine".
                    long_enough = words >= 2 or len(text) >= 10
                else:
                    long_enough = words >= 2 or len(text) >= 8

                # Avoid aggressive one-word finals such as "med" while user is
                # likely still forming the utterance.
                if not long_enough and (now - pending_partial_at) < short_fallback_after_s:
                    logging.info("Fallback suppressed (partial too short): %s", text)
                    continue

                # If phrase ends with a connector ("... i", "... og"), wait
                # longer before finalizing to avoid splitting the next word.
                if _looks_incomplete_phrase(text) and (now - pending_partial_at) < (short_fallback_after_s + 1.5):
                    logging.info("Fallback suppressed (partial looks incomplete): %s", text)
                    continue

                if cfg.provider == "azure" and cfg.audio_source == "browser" and _should_block_short_final(text):
                    blocked_for_s = now - pending_partial_at
                    logging.info("Fallback suppressed (short/incomplete final): %s", text)
                    # Drop stale, never-completing fragments after a long wait.
                    if blocked_for_s > 20.0:
                        logging.info("Pending partial dropped (stale short/incomplete): %s", text)
                        pending_partial = ""
                    continue

                pending_partial = ""
                await emit_final(text, "fallback_from_partial")
            continue

        if kind == "partial":
            text = (text or "").strip()
            if cfg.provider == "azure" and cfg.audio_source == "browser":
                if last_final_text:
                    text = _trim_repeated_prefix(text, last_final_text)
                text = _trim_known_prefix(text)
            if text:
                pending_partial = text
                pending_partial_at = time.time()
                await broadcast({"partial": text})
            continue

        if kind == "final":
            now = time.time()
            text_clean = (text or "").strip()
            if cfg.provider == "azure" and cfg.audio_source == "browser":
                if last_final_text:
                    text_clean = _trim_repeated_prefix(text_clean, last_final_text)
                text_clean = _trim_known_prefix(text_clean)
            if _is_noise_fragment(text_clean):
                logging.info("Provider final suppressed (noise fragment): %s", text_clean)
                if text_clean:
                    pending_partial = text_clean
                    pending_partial_at = now
                continue
            if cfg.provider == "azure" and cfg.audio_source == "browser" and _should_block_short_final(text_clean):
                logging.info("Provider final suppressed (short/incomplete): %s", text_clean)
                if text_clean:
                    pending_partial = text_clean
                    pending_partial_at = now
                continue
            ingest_recently_active = (now - LAST_INGEST_AT) < ingest_idle_before_finalize_s

            # Azure can sometimes emit early finals mid-utterance.
            # If ingest is still active, treat finals as partial and wait for
            # real silence before emitting final.
            if (
                cfg.provider == "azure"
                and cfg.audio_source == "browser"
                and ingest_recently_active
            ):
                words = len(text_clean.split())
                maybe_early = words <= 1 or len(text_clean) < 10
                if maybe_early:
                    pending_partial = text_clean
                    pending_partial_at = now
                    if text_clean:
                        await broadcast({"partial": text_clean})
                    logging.info("Final downgraded to partial (azure-active-ingest): %s", text_clean)
                    continue

            pending_partial = ""
            await emit_final(text_clean, "provider")


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
        except Exception as exc:
            logging.warning("Ingest client error: %s", exc)
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
    global INGEST_QUEUE, AZURE_STT_RESET_EVENT

    cfg = parse_args()
    log_handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.log_file.strip():
        log_handlers.append(logging.FileHandler(cfg.log_file.strip(), encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=log_handlers,
        force=True,
    )

    if cfg.list_input_devices:
        print_input_devices()
        return

    INGEST_QUEUE = asyncio.Queue(maxsize=512)
    AZURE_STT_RESET_EVENT = asyncio.Event()
    event_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=512)

    logging.info(
        "Starting transcriber provider=%s source=%s on %s:%s (ws: /ingest + /transcribe, http: /final) azure_segmentation_silence_ms=%s",
        cfg.provider,
        cfg.audio_source,
        cfg.host,
        cfg.port,
        cfg.azure_segmentation_silence_ms,
    )
    if cfg.provider == "azure" and cfg.audio_source == "browser":
        logging.info(
            "Azure fallback-final policy: guarded-enabled (idle>=1.6s, min_words>=2 or len>=10)"
        )
    else:
        logging.info(
            "Azure fallback-final policy: %s",
            "enabled" if cfg.azure_allow_fallback_final else "disabled",
        )

    async with serve(ws_handler, cfg.host, cfg.port, process_request=process_request, max_size=2_000_000):
        await asyncio.gather(provider_loop(cfg, event_queue), event_broadcast_loop(cfg, event_queue))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
