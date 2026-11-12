#!/usr/bin/env bash
# First-time setup for macOS and Linux.
# Usage:  ./setup.sh            (run from the project folder)
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
# The product name lives in brand.json; nothing else in this script hard-codes it.
APP_NAME="$(sed -n 's/.*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' brand.json 2>/dev/null | head -n 1)"
APP_NAME="${APP_NAME:-DocLens}"

say()  { printf '\033[36m%s\033[0m\n' "$*"; }
ok()   { printf '\033[32m%s\033[0m\n' "$*"; }
fail() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

say "$APP_NAME first-time setup"

command -v docker >/dev/null 2>&1 || fail "Docker is not installed. Install Docker Desktop, start it, and run ./setup.sh again."
command -v npm    >/dev/null 2>&1 || fail "Node.js is not installed. Install Node.js 20.19 or newer (https://nodejs.org) and run ./setup.sh again."

if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (fast Python package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -f .env ]; then
  cp .env.example .env
  ok "Created .env from the safe template."
fi
if grep -Eq '^GEMINI_API_KEY=\s*$' .env; then
  printf 'Paste your Gemini API key (from https://aistudio.google.com/apikey): '
  read -r key
  [ -n "$key" ] || fail "No key entered. Add GEMINI_API_KEY to .env and run ./setup.sh again."
  # replace the empty line in place (portable between GNU and BSD sed)
  tmp="$(mktemp)"; sed "s|^GEMINI_API_KEY=.*|GEMINI_API_KEY=$key|" .env > "$tmp" && mv "$tmp" .env
fi

say "Creating the Python 3.12 environment and installing backend packages..."
cd "$ROOT/backend"
# --relocatable keeps the environment working if the project folder is renamed or moved
[ -x .venv/bin/python ] || uv venv .venv --python 3.12 --relocatable
# --torch-backend=auto picks the right PyTorch build (Apple MPS on Mac, CUDA on NVIDIA Linux)
uv pip install --python .venv/bin/python --torch-backend=auto -r requirements.txt

say "Installing frontend packages..."
cd "$ROOT/frontend"
npm ci

say "Downloading the AI models once (about 2 GB; later runs reuse them)..."
cd "$ROOT/backend"
.venv/bin/python -m scripts.download_models

ok "Setup complete. Start $APP_NAME with ./start.sh"
