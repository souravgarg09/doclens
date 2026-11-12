#!/usr/bin/env bash
# Stop DocLens on macOS or Linux. Stored documents and answers are kept.
# Usage:  ./stop.sh   (add --keep-infrastructure to leave the Docker databases running)
set -uo pipefail

cd "$(dirname "$0")"
PIDS=".runtime/pids"

for name in frontend worker api; do
  file="$PIDS/$name"
  [ -f "$file" ] || continue
  pid="$(cat "$file")"
  # only stop the process if it is still a DocLens process (PIDs get reused after a reboot)
  cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  if [ -n "$cmd" ] && printf '%s' "$cmd" | grep -Eq 'uvicorn|parse_worker|npm|vite|node'; then
    # stop the process and its children (npm starts vite as a child process)
    pkill -TERM -P "$pid" 2>/dev/null
    kill -TERM "$pid" 2>/dev/null
    echo "Stopped $name."
  fi
  rm -f "$file"
done

if [ "${1:-}" != "--keep-infrastructure" ] && command -v docker >/dev/null 2>&1; then
  docker compose --project-directory "$(pwd)" stop && echo "Stopped Docker services. Stored data was preserved."
fi
printf '\033[32m%s\033[0m\n' "DocLens is stopped."
