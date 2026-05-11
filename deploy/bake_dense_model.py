"""Snapshot-download the dense-retrieval model into the image at build time.

Run from the Dockerfile with HF_REPO and HF_TARGET set. The image's
bind-mount target stays the same; this just guarantees a usable model
inside the image even when no host bind mount is provided. Failure exits
non-zero so the build fails loudly rather than producing an image that
crashes at first encode.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    repo = os.environ.get("HF_REPO", "").strip()
    target = os.environ.get("HF_TARGET", "").strip()
    if not repo or not target:
        print("HF_REPO and HF_TARGET must both be set", file=sys.stderr)
        return 2

    print(f"Baking dense model {repo} -> {target}", flush=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print(f"huggingface_hub not available: {exc}", file=sys.stderr)
        return 3

    try:
        snapshot_download(
            repo_id=repo,
            local_dir=target,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.bin",
                "*.safetensors",
                "*.model",
                "tokenizer*",
                "sentence_bert_config.json",
                "modules.json",
                "config_sentence_transformers.json",
                "1_Pooling/*",
            ],
        )
    except Exception as exc:
        print(f"snapshot_download failed: {exc}", file=sys.stderr)
        return 4

    print(f"Dense model baked at {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
