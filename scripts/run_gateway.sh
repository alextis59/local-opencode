#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"
exec python scripts/serve_gateway.py
