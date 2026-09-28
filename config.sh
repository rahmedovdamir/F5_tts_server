# Shared by start_server.sh / stop_server.sh. Every value can be overridden via environment.
SERVER_DIR="${F5TTS_SERVER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${F5TTS_PYTHON_BIN:-$HOME/f5tts-venv/bin/python}"
LOG_FILE="${F5TTS_LOG_FILE:-$HOME/f5tts-server-v2.log}"
PID_FILE="${F5TTS_PID_FILE:-$HOME/f5tts-server-v2.pid}"
PORT="${F5TTS_PORT:-8888}"
