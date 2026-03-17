#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

stop_pid_file() {
  local pid_file="$1"
  local name="$2"

  if [[ ! -f "$pid_file" ]]; then
    echo "$name: no pid file"
    return
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    echo "$name: empty pid file"
    rm -f "$pid_file"
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 0.5
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    echo "$name: stopped pid $pid"
  else
    echo "$name: pid $pid not running"
  fi

  rm -f "$pid_file"
}

stop_pid_file "logs/transcriber.pid" "transcriber"
stop_pid_file "logs/streamlit.pid" "streamlit"

# Best-effort cleanup for stale processes not tracked by pid files.
pkill -f "realtime_transcriber.py" >/dev/null 2>&1 || true
pkill -f "streamlit run app.py" >/dev/null 2>&1 || true
