# local-opencode

Local OpenAI-compatible gateway for running
[WeiboAI/VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B)
behind [OpenCode](https://opencode.ai/docs/providers/).

The default runtime path uses a CPU-friendly GGUF quantization:
[`oussaber/VibeThinker-3B-Q4_K_M-GGUF`](https://huggingface.co/oussaber/VibeThinker-3B-Q4_K_M-GGUF).
The original model is MIT licensed, Qwen2-based, and distributed as BF16
safetensors. The GGUF keeps the same chat template metadata and is small enough
to run on machines without a working NVIDIA driver.

## Setup

On Ubuntu, the one-command installer is:

```bash
scripts/install_ubuntu.sh
```

It installs Ubuntu build dependencies, creates `.venv`, installs Python
packages, installs `opencode-ai` through npm if needed, downloads the GGUF, and
writes `scripts/run_gateway.sh`. To start the gateway as part of the install:

```bash
scripts/install_ubuntu.sh --start
```

Use `scripts/install_ubuntu.sh --help` for options such as `--no-opencode`,
`--skip-model`, and `--copy-model`.

Manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py
```

On Ubuntu/Debian, install `python3.10-venv` first if `python3 -m venv`
reports that `ensurepip` is unavailable. If you cannot install the venv package,
use the user site instead:

```bash
python3 -m pip install --user -r requirements.txt
python3 scripts/download_model.py
```

`llama-cpp-python` may build from source. This machine used `cmake`, `make`, and
`g++` and built the CPU wheel successfully.

By default the downloader creates `models/vibethinker-3b-q4_k_m.gguf` as a
symlink into the Hugging Face cache. Pass `--copy` if you want an independent
copy in this repo.

## Run the gateway

```bash
source .venv/bin/activate
python scripts/serve_gateway.py
```

If you used the installer, run:

```bash
scripts/run_gateway.sh
```

The gateway listens on `http://127.0.0.1:8088` and exposes:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`

Smoke test:

```bash
curl http://127.0.0.1:8088/v1/models
curl -s http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"vibethinker-3b","messages":[{"role":"user","content":"What is 84 * 3 / 2?"}],"max_tokens":64}' \
  | python3 -m json.tool
```

## OpenCode

This repo includes a project-level `opencode.json` that registers:

- provider: `vibethinker-local`
- model: `vibethinker-3b`
- base URL: `http://127.0.0.1:8088/v1`

Start the gateway, then run OpenCode from this repository:

```bash
opencode
```

If OpenCode asks for credentials for `vibethinker-local`, use any non-empty API
key. The local gateway does not validate it.

## Runtime knobs

Copy `.env.example` if you want shell-managed settings. The most useful values:

- `VIBETHINKER_N_CTX=8192`: context size. Increase to `16384` or `32768` if you have
  enough RAM and need more project context.
- `VIBETHINKER_N_THREADS=0`: llama.cpp auto-selects CPU threads. Set an explicit
  count if you want tighter CPU control.
- `VIBETHINKER_N_GPU_LAYERS=0`: CPU-only. Set higher if you have llama.cpp GPU
  support available.
- `VIBETHINKER_DEFAULT_MAX_TOKENS=64`: fallback output length when callers do
  not send `max_tokens`.
- `VIBETHINKER_MAX_TOKENS=64`: hard output cap. The default is intentionally
  low so OpenCode smoke tests complete on CPU-only machines. Raise it to `256`
  or `512` for better answers, or set it to `0` to accept the caller's requested
  output length.
- `VIBETHINKER_BUFFER_STREAM=true`: buffers streamed requests so the gateway can
  strip `<think>` tags and convert XML tool calls before OpenCode sees them.
- `VIBETHINKER_FALLBACK_TO_REASONING=true`: if the model is cut off before it
  leaves its thinking block, return the sanitized reasoning text as content
  instead of an empty assistant message.
- `VIBETHINKER_FORWARD_TOOLS=false`: skip forwarding OpenCode's full tool schema
  into llama.cpp. Set to `true` when you want to experiment with local tool
  calling and can tolerate the much larger prompt.

## Notes

The gateway passes OpenAI-compatible chat requests to `llama-cpp-python` and
normalizes VibeThinker/Qwen XML tool-call blocks into OpenAI-style `tool_calls`.
That normalization is needed for OpenCode because local models often emit the
tool call text from the chat template rather than structured API fields.
