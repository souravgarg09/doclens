#!/usr/bin/env bash
# Start the application on macOS or Linux: databases (Docker), API, parsing worker, frontend.
# Usage:  ./start.sh            (add --no-browser to skip opening the browser)
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
# The product name lives in brand.json; nothing else in this script hard-codes it.
APP_NAME="$(sed -n 's/.*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' brand.json 2>/dev/null | head -n 1)"
APP_NAME="${APP_NAME:-DocLens}"
RUNTIME="$ROOT/.runtime"
LOGS="$RUNTIME/logs"
PIDS="$RUNTIME/pids"
PY="$ROOT/backend/.venv/bin/python"

say()  { printf '\033[36m%s\033[0m\n' "$*"; }
ok()   { printf '\033[32m%s\033[0m\n' "$*"; }
fail() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
wait_port() { local port=$1 secs=$2; for _ in $(seq 1 $((secs * 2))); do port_open "$port" && return 0; sleep 0.5; done; return 1; }
wait_http() { local url=$1 secs=$2; for _ in $(seq 1 $((secs * 2))); do curl -fs -o /dev/null "$url" && return 0; sleep 0.5; done; return 1; }

[ -x "$PY" ] || fail "Backend environment is missing. Run ./setup.sh first."
[ -f .env ] || fail ".env is missing. Run ./setup.sh first."
[ -d frontend/node_modules ] || fail "Frontend packages are missing. Run ./setup.sh first."
for p in 8000 5173; do
  port_open "$p" && fail "Port $p is already in use. Run ./stop.sh first if $APP_NAME is already running."
done

if ! docker info >/dev/null 2>&1; then
  if [ "$(uname)" = "Darwin" ]; then say "Starting Docker Desktop..."; open -a Docker; fi
  for _ in $(seq 1 90); do docker info >/dev/null 2>&1 && break; sleep 1; done
  docker info >/dev/null 2>&1 || fail "Docker did not start within 90 seconds. Open Docker Desktop and try again."
fi

say "Starting PostgreSQL, Redis and Qdrant..."
docker compose --project-directory "$ROOT" up -d
for p in 5432 6379 6333; do wait_port "$p" 90 || fail "Database service on port $p did not start."; done

mkdir -p "$LOGS" "$PIDS"
say "Starting API, parsing worker and frontend..."
( cd backend && nohup "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 >"$LOGS/api.log" 2>&1 & echo $! >"$PIDS/api" )
( cd backend && nohup "$PY" -m workers.parse_worker >"$LOGS/worker.log" 2>&1 & echo $! >"$PIDS/worker" )
( cd frontend && nohup npm run dev -- --host 127.0.0.1 >"$LOGS/frontend.log" 2>&1 & echo $! >"$PIDS/frontend" )

wait_http "http://127.0.0.1:8000/api/health" 180 || { ./stop.sh --keep-infrastructure; fail "The API did not start. See .runtime/logs/api.log"; }
wait_http "http://127.0.0.1:5173/" 60 || { ./stop.sh --keep-infrastructure; fail "The frontend did not start. See .runtime/logs/frontend.log"; }

ok "$APP_NAME is ready: http://127.0.0.1:5173/"
echo "The worker loads its AI models in the background; the page shows when it is ready."
echo "Logs: $LOGS    Stop with: ./stop.sh"
if [ "${1:-}" != "--no-browser" ]; then
  if command -v open >/dev/null 2>&1; then open "http://127.0.0.1:5173/"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "http://127.0.0.1:5173/"; fi
fi
