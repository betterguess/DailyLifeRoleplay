#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing .venv python at $VENV_PY"
  echo "Create it first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt"
  exit 1
fi

mkdir -p logs
rm -f logs/app-speech.log logs/transcriber.log logs/streamlit.log logs/transcriber.pid logs/streamlit.pid

# Optional env defaults for local browser->backend speech flow.
export TRANSCRIBER_WS="${TRANSCRIBER_WS:-ws://localhost:9000/transcribe}"
export TRANSCRIBER_INGEST_WS="${TRANSCRIBER_INGEST_WS:-ws://localhost:9000/ingest}"
export APP_SPEECH_LOG_FILE="${APP_SPEECH_LOG_FILE:-logs/app-speech.log}"
export STT_LOG_FILE="${STT_LOG_FILE:-logs/transcriber.log}"
export MIC_SILENCE_RMS="${MIC_SILENCE_RMS:-0.006}"
export DEBUG_MIC="${DEBUG_MIC:-false}"

# Start transcriber (Azure/browser defaults can be overridden via env or args here).
nohup "$VENV_PY" realtime_transcriber.py \
  --provider "${STT_PROVIDER:-azure}" \
  --audio-source "${STT_AUDIO_SOURCE:-browser}" \
  --azure-language "${AZURE_SPEECH_LANGUAGE:-da-DK}" \
  --azure-segmentation-silence-ms "${AZURE_SEGMENTATION_SILENCE_MS:-2200}" \
  --log-file "$STT_LOG_FILE" \
  >> logs/transcriber.log 2>&1 &
TRANSCRIBER_PID=$!
echo "$TRANSCRIBER_PID" > logs/transcriber.pid

# Start Streamlit app.
nohup "$VENV_PY" -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 \
  >> logs/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "$STREAMLIT_PID" > logs/streamlit.pid

sleep 1

echo "Started services:"
echo "- transcriber pid: $TRANSCRIBER_PID (log: logs/transcriber.log)"
echo "- streamlit pid:   $STREAMLIT_PID (log: logs/streamlit.log)"
echo
echo "Useful commands:"
echo "- tail -f logs/transcriber.log"
echo "- tail -f logs/streamlit.log"
echo "- ./scripts/stop_services.sh"
