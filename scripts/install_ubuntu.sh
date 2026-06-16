#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
LOG_FILE="${LOG_FILE:-/tmp/local-opencode-vibethinker.log}"
NODE_MAJOR="${NODE_MAJOR:-22}"

INSTALL_APT=1
INSTALL_OPENCODE=1
DOWNLOAD_MODEL=1
COPY_MODEL=0
START_GATEWAY=0

usage() {
  cat <<'EOF'
Usage: scripts/install_ubuntu.sh [options]

Installs the local VibeThinker gateway dependencies on Ubuntu, downloads the
GGUF model, and optionally starts the gateway.

Options:
  --no-apt             Skip apt package installation.
  --no-opencode        Skip npm install -g opencode-ai.
  --skip-model         Skip Hugging Face model download.
  --copy-model         Copy the GGUF into ./models instead of symlinking the HF cache.
  --start              Start the gateway in the background after installation.
  -h, --help           Show this help.

Environment:
  VENV_DIR             Python virtualenv path. Default: ./.venv
  NODE_MAJOR           NodeSource major version if Node must be installed. Default: 22
  LOG_FILE             Gateway log path when using --start. Default: /tmp/local-opencode-vibethinker.log
EOF
}

log() {
  printf '\n==> %s\n' "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

sudo_cmd() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

node_major() {
  if ! have node; then
    echo 0
    return
  fi
  node --version | sed -E 's/^v([0-9]+).*/\1/'
}

install_apt_packages() {
  log "Installing Ubuntu build/runtime packages"

  sudo_cmd apt-get update
  sudo_cmd apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    g++ \
    git \
    make \
    ninja-build \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv

  if ! have node || [[ "$(node_major)" -lt 20 ]]; then
    log "Installing Node.js ${NODE_MAJOR}.x from NodeSource"
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | sudo_cmd bash -
    sudo_cmd apt-get install -y nodejs
  fi
}

install_python_deps() {
  log "Creating Python virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR" || die "python3 -m venv failed. Install python3-venv, then rerun this script."

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  log "Installing Python dependencies"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "$ROOT_DIR/requirements.txt"
}

install_opencode() {
  if have opencode; then
    log "OpenCode already installed: $(opencode --version)"
    return
  fi

  have npm || die "npm is not installed. Rerun without --no-apt or install Node.js/npm first."

  log "Installing OpenCode with npm"
  local npm_prefix
  npm_prefix="$(npm config get prefix)"

  if [[ "${EUID}" -eq 0 || -w "$npm_prefix" ]]; then
    npm install -g opencode-ai
  else
    sudo env "PATH=$PATH" npm install -g opencode-ai
  fi
}

download_model() {
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  log "Downloading VibeThinker-3B GGUF model"
  if [[ "$COPY_MODEL" -eq 1 ]]; then
    python "$ROOT_DIR/scripts/download_model.py" --copy
  else
    python "$ROOT_DIR/scripts/download_model.py"
  fi
}

write_runner() {
  log "Writing helper runner: scripts/run_gateway.sh"
  cat >"$ROOT_DIR/scripts/run_gateway.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="\${VENV_DIR:-$VENV_DIR}"
source "\${VENV_DIR}/bin/activate"
cd "\${ROOT_DIR}"
exec python scripts/serve_gateway.py
EOF
  chmod +x "$ROOT_DIR/scripts/run_gateway.sh"
}

start_gateway() {
  log "Starting gateway in the background"
  pkill -f "$ROOT_DIR/scripts/serve_gateway.py" 2>/dev/null || true
  setsid bash -lc "cd '$ROOT_DIR' && '$ROOT_DIR/scripts/run_gateway.sh' >'$LOG_FILE' 2>&1" >/dev/null 2>&1 &
  sleep 2

  if ! curl -fsS http://127.0.0.1:8088/healthz >/dev/null; then
    printf 'Gateway failed to start. Recent log:\n' >&2
    tail -80 "$LOG_FILE" >&2 || true
    exit 1
  fi

  log "Gateway is running at http://127.0.0.1:8088"
  printf 'Log: %s\n' "$LOG_FILE"
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-apt)
        INSTALL_APT=0
        ;;
      --no-opencode)
        INSTALL_OPENCODE=0
        ;;
      --skip-model)
        DOWNLOAD_MODEL=0
        ;;
      --copy-model)
        COPY_MODEL=1
        ;;
      --start)
        START_GATEWAY=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
    shift
  done

  [[ -f "$ROOT_DIR/requirements.txt" ]] || die "Run this script from the local-opencode repo checkout."
  [[ -f "$ROOT_DIR/opencode.json" ]] || die "opencode.json is missing from the repo checkout."

  if [[ "$INSTALL_APT" -eq 1 ]]; then
    have apt-get || die "This installer is for Ubuntu/Debian systems with apt-get."
    install_apt_packages
  fi

  install_python_deps

  if [[ "$INSTALL_OPENCODE" -eq 1 ]]; then
    install_opencode
  fi

  if [[ "$DOWNLOAD_MODEL" -eq 1 ]]; then
    download_model
  fi

  write_runner

  log "Install complete"
  cat <<EOF
Next steps:
  1. Start the gateway:
       scripts/run_gateway.sh

  2. In another terminal, from this repo:
       opencode

  3. Use model:
       vibethinker-local/vibethinker-3b

Smoke test:
  curl http://127.0.0.1:8088/healthz
EOF

  if [[ "$START_GATEWAY" -eq 1 ]]; then
    start_gateway
  fi
}

main "$@"
