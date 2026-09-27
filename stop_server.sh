#!/usr/bin/env bash
set -euo pipefail

PID_FILE="/home/rdr/f5tts-server-v2.pid"

stopped=0
if [[ -f "$PID_FILE" ]]; then
    server_pid="$(cat "$PID_FILE")"
    if kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        echo "F5-TTS server stopped (PID $server_pid)"
        stopped=1
    fi
fi
rm -f "$PID_FILE"

# Also stop an older instance that was launched before PID tracking existed.
while read -r server_pid; do
    [[ -z "$server_pid" ]] && continue
    if kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        echo "Old F5-TTS server stopped (PID $server_pid)"
        stopped=1
    fi
done < <(pgrep -f '^/home/rdr/f5tts-venv/bin/python -m uvicorn server:app --host 0\.0\.0\.0 --port 8888$' || true)

if [[ "$stopped" -eq 0 ]]; then
    echo "F5-TTS server is not running"
fi
