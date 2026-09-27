#!/usr/bin/env bash
set -euo pipefail

SERVER_DIR="/home/rdr/F5-TTS_server"
PYTHON_BIN="/home/rdr/f5tts-venv/bin/python"
LOG_FILE="/home/rdr/f5tts-server-v2.log"
PID_FILE="/home/rdr/f5tts-server-v2.pid"

cd "$SERVER_DIR"

port_pids="$(fuser -n tcp 8888 2>/dev/null || true)"
if [[ -n "$port_pids" ]]; then
    echo "Port 8888 is already occupied by PID(s):$port_pids"
    echo "Run: $SERVER_DIR/stop_server.sh"
    exit 1
fi

if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE")"
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Server is already running (PID $old_pid)"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

PYTHONPATH="$SERVER_DIR${PYTHONPATH:+:$PYTHONPATH}" \
nohup "$PYTHON_BIN" -m uvicorn server:app \
    --host 0.0.0.0 --port 8888 \
    > "$LOG_FILE" 2>&1 &

server_pid=$!
echo "$server_pid" > "$PID_FILE"
disown "$server_pid" 2>/dev/null || true
echo "F5-TTS server started (PID $server_pid)"
echo "Log: $LOG_FILE"
