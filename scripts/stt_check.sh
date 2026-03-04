#!/usr/bin/env bash
set -euo pipefail

missing=()
for key in AZURE_SPEECH_KEY AZURE_SPEECH_REGION; do
  if [[ -z "${!key:-}" ]]; then
    missing+=("$key")
  fi
done

if ((${#missing[@]} > 0)); then
  echo "Missing env: ${missing[*]}"
else
  echo "Missing env: none"
fi

host="${STT_HOST:-127.0.0.1}"
port="${STT_PORT:-9000}"

if command -v nc >/dev/null 2>&1; then
  if nc -z -w 2 "$host" "$port" >/dev/null 2>&1; then
    echo "Port ${host}:${port} open: true"
  else
    echo "Port ${host}:${port} open: false"
  fi
else
  python3 - "$host" "$port" <<'PY'
import socket
import sys
host = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket()
s.settimeout(1.5)
res = s.connect_ex((host, port))
s.close()
print(f"Port {host}:{port} open: {res == 0}")
PY
fi

if ((${#missing[@]} > 0)); then
  exit 1
fi
