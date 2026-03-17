<script type="module">
const SHOULD_LISTEN = __LISTENING__;
const INGEST_WS = "__INGEST_WS__";
const DEBUG_MIC = __DEBUG_MIC__;
const TARGET_SR = 16000;
const SILENCE_RMS_THRESHOLD = Number(__MIC_SILENCE_RMS__) || 0;
const root = (window.parent && window.parent.window) ? window.parent.window : window;
const LOG_PREFIX = "[mic]";

function resolveIngestWs() {
  if (INGEST_WS && INGEST_WS !== "auto") return INGEST_WS;
  const proto = root.location && root.location.protocol === "https:" ? "wss" : "ws";
  const host = (root.location && root.location.hostname) ? root.location.hostname : "localhost";
  return `${proto}://${host}:9000/ingest`;
}

function logInfo(message, extra) {
  if (!DEBUG_MIC) return;
  if (extra !== undefined) {
    console.info(`${LOG_PREFIX} ${message}`, extra);
  } else {
    console.info(`${LOG_PREFIX} ${message}`);
  }
}

function logError(message, error) {
  if (!DEBUG_MIC) return;
  console.error(`${LOG_PREFIX} ${message}`, error);
}

function downsampleTo16k(float32Array, sourceRate) {
  if (sourceRate === TARGET_SR) return float32Array;

  const ratio = sourceRate / TARGET_SR;
  const newLength = Math.round(float32Array.length / ratio);
  const result = new Float32Array(newLength);

  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;

    for (let i = offsetBuffer; i < nextOffsetBuffer && i < float32Array.length; i++) {
      accum += float32Array[i];
      count += 1;
    }

    result[offsetResult] = count > 0 ? (accum / count) : 0;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
}

function toInt16Buffer(float32Array) {
  const pcm16 = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const sample = Math.max(-1, Math.min(1, float32Array[i]));
    pcm16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return pcm16.buffer;
}

function rms(samples) {
  if (!samples || samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    sum += s * s;
  }
  return Math.sqrt(sum / samples.length);
}

function ensureMicState() {
  if (!root.__dlrMic) {
    root.__dlrMic = {
      started: false,
      desiredListening: false,
      ws: null,
      stream: null,
      audioCtx: null,
      source: null,
      processor: null,
      connecting: false,
      watchdogTimer: null,
      lastProcessAt: 0,
      lastSentAt: 0,
      lastVoiceAt: 0,
    };
  }
  return root.__dlrMic;
}

function closeWebSocket(state) {
  try {
    if (state.ws) state.ws.close();
  } catch (error) {
    // ignore close errors
  }
  state.ws = null;
}

function stopMic(state) {
  state.started = false;

  try { if (state.processor) state.processor.disconnect(); } catch (error) {}
  try { if (state.source) state.source.disconnect(); } catch (error) {}
  try { if (state.stream) state.stream.getTracks().forEach((track) => track.stop()); } catch (error) {}
  try { if (state.audioCtx) state.audioCtx.close(); } catch (error) {}

  state.processor = null;
  state.source = null;
  state.stream = null;
  state.audioCtx = null;

  closeWebSocket(state);
}

function wireProcessor(state, audioCtx, processor) {
  processor.onaudioprocess = (event) => {
    try {
      state.lastProcessAt = Date.now();
      const wsOpen = state.ws && state.ws.readyState === root.WebSocket.OPEN;
      if (!state.started || !wsOpen) return;

      const input = event.inputBuffer.getChannelData(0);
      const downsampled = downsampleTo16k(input, audioCtx.sampleRate);
      const level = rms(downsampled);
      const voiceMarker = SILENCE_RMS_THRESHOLD > 0 ? SILENCE_RMS_THRESHOLD * 1.2 : 0.012;
      if (level >= voiceMarker) {
        state.lastVoiceAt = Date.now();
      }

      if (SILENCE_RMS_THRESHOLD > 0 && level < SILENCE_RMS_THRESHOLD) {
        return;
      }
      const pcm = toInt16Buffer(downsampled);
      state.ws.send(pcm);
      state.lastSentAt = Date.now();
    } catch (error) {
      // ignore transient audio/ws errors
    }
  };
}

async function openIngestSocket() {
  const targetWs = resolveIngestWs();
  logInfo("opening websocket", { url: targetWs });
  const ws = new root.WebSocket(targetWs);
  ws.binaryType = "arraybuffer";
  ws.onerror = (event) => {
    logError("websocket error", event);
  };

  await new Promise((resolve, reject) => {
    ws.onopen = () => resolve();
    ws.onerror = () => reject(new Error("ws open failed"));
  });

  return ws;
}

async function createAudioGraph() {
  const stream = await root.navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  const audioCtx = new (root.AudioContext || root.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);
  const mutedGain = audioCtx.createGain();
  mutedGain.gain.value = 0;

  source.connect(processor);
  processor.connect(mutedGain);
  mutedGain.connect(audioCtx.destination);

  return { stream, audioCtx, source, processor };
}

async function startMic(state) {
  if (!state.desiredListening) return;
  if (state.started || state.connecting) return;

  state.connecting = true;
  logInfo("startMic attempt");
  try {
    const { stream, audioCtx, source, processor } = await createAudioGraph();
    const ws = await openIngestSocket();

    state.ws = ws;
    logInfo("websocket connected");

    state.stream = stream;
    state.audioCtx = audioCtx;
    state.source = source;
    state.processor = processor;
    state.started = true;
    state.lastProcessAt = Date.now();
    state.lastSentAt = Date.now();
    state.lastVoiceAt = 0;

    wireProcessor(state, audioCtx, processor);

    ws.onclose = () => {
      const wasStarted = state.started;
      logInfo("websocket closed", { wasStarted, desiredListening: state.desiredListening });
      if (!wasStarted) return;

      stopMic(state);
      if (state.desiredListening) {
        logInfo("reconnect scheduled");
        root.setTimeout(() => startMic(state), 400);
      }
    };
  } catch (error) {
    logError("startMic failed", error);
    stopMic(state);
  } finally {
    state.connecting = false;
  }
}

function ensureWatchdog(state) {
  // Streamlit re-renders this component frequently. Rebind watchdog each
  // render so callbacks always point at the current script instance.
  if (state.watchdogTimer) {
    try {
      root.clearInterval(state.watchdogTimer);
    } catch (error) {
      // ignore timer cleanup errors
    }
    state.watchdogTimer = null;
  }

  state.watchdogTimer = root.setInterval(() => {
    if (state.desiredListening) {
      const wsOpen = state.ws && state.ws.readyState === root.WebSocket.OPEN;
      const streamTracks = state.stream ? state.stream.getAudioTracks() : [];
      const micLive = streamTracks.some((track) => track.readyState === "live");
      const now = Date.now();
      const stalledProcessor = state.started && state.lastProcessAt > 0 && (now - state.lastProcessAt) > 3000;
      const staleTransport = (
        state.started
        && state.lastVoiceAt > 0
        && (now - state.lastVoiceAt) < 4000
        && state.lastSentAt > 0
        && (now - state.lastSentAt) > 4000
      );
      const audioSuspended = state.audioCtx && state.audioCtx.state && state.audioCtx.state !== "running";

      if (audioSuspended && state.audioCtx && typeof state.audioCtx.resume === "function") {
        state.audioCtx.resume().catch(() => {});
      }

      if (!state.started || !wsOpen || !micLive || stalledProcessor || staleTransport) {
        logInfo("watchdog restart", {
          started: state.started,
          wsOpen: Boolean(wsOpen),
          micLive,
          stalledProcessor,
          staleTransport,
          audioState: state.audioCtx ? state.audioCtx.state : "none",
          msSinceProcess: state.lastProcessAt ? now - state.lastProcessAt : null,
          msSinceSent: state.lastSentAt ? now - state.lastSentAt : null,
          trackCount: streamTracks.length,
        });
        if (state.started) {
          stopMic(state);
        }
        startMic(state);
      }
      return;
    }

    if (state.started) {
      stopMic(state);
    }
  }, 1000);
}

function main() {
  const state = ensureMicState();
  state.desiredListening = SHOULD_LISTEN;

  if (state.desiredListening) {
    startMic(state);
  } else {
    stopMic(state);
  }

  ensureWatchdog(state);
}

main();
</script>
