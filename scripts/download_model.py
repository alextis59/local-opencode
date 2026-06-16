#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO = "oussaber/VibeThinker-3B-Q4_K_M-GGUF"
DEFAULT_FILENAME = "vibethinker-3b-q4_k_m.gguf"
DEFAULT_OUTPUT = "models/vibethinker-3b-q4_k_m.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the VibeThinker-3B GGUF used by the gateway.")
    parser.add_argument("--repo", default=os.getenv("VIBETHINKER_HF_REPO", DEFAULT_REPO))
    parser.add_argument("--filename", default=os.getenv("VIBETHINKER_HF_FILENAME", DEFAULT_FILENAME))
    parser.add_argument("--output", default=os.getenv("VIBETHINKER_MODEL_PATH", DEFAULT_OUTPUT))
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the file out of the Hugging Face cache instead of creating a symlink.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    source = Path(hf_hub_download(repo_id=args.repo, filename=args.filename))
    if output.exists() or output.is_symlink():
        output.unlink()

    if args.copy:
        shutil.copy2(source, output)
    else:
        output.symlink_to(source)

    print(f"Model ready: {output} -> {source}")


if __name__ == "__main__":
    main()
