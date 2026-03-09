<script type="module">
const SHOULD_LISTEN = __LISTENING__;
const INGEST_WS = "__INGEST_WS__";
const TARGET_SR = 16000;
const root = (window.parent && window.parent.window) ? window.parent.window : window;

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
      const wsOpen = state.ws && state.ws.readyState === root.WebSocket.OPEN;
      if (!state.started || !wsOpen) return;

      const input = event.inputBuffer.getChannelData(0);
      const downsampled = downsampleTo16k(input, audioCtx.sampleRate);
      const pcm = toInt16Buffer(downsampled);
      state.ws.send(pcm);
    } catch (error) {
      // ignore transient audio/ws errors
    }
  };
}

async function openIngestSocket() {
  const ws = new root.WebSocket(INGEST_WS);
  ws.binaryType = "arraybuffer";

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
  try {
    const ws = await openIngestSocket();
    state.ws = ws;

    const { stream, audioCtx, source, processor } = await createAudioGraph();
    wireProcessor(state, audioCtx, processor);

    state.stream = stream;
    state.audioCtx = audioCtx;
    state.source = source;
    state.processor = processor;
    state.started = true;

    ws.onclose = () => {
      const wasStarted = state.started;
      if (!wasStarted) return;

      stopMic(state);
      if (state.desiredListening) {
        root.setTimeout(() => startMic(state), 400);
      }
    };
  } catch (error) {
    stopMic(state);
  } finally {
    state.connecting = false;
  }
}

function ensureWatchdog(state) {
  if (state.watchdogTimer) return;

  state.watchdogTimer = root.setInterval(() => {
    if (state.desiredListening) {
      const wsOpen = state.ws && state.ws.readyState === root.WebSocket.OPEN;
      if (!state.started || !wsOpen) {
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
