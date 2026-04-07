from __future__ import annotations

import importlib.util
from pathlib import Path


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    print(f"project_root={Path(__file__).resolve().parents[1]}")
    modules = ["spacy", "textacy", "stanza", "nltk", "gensim", "flair", "textblob", "torch"]
    for name in modules:
        print(f"{name}={_module_available(name)}")

    try:
        import torch

        print(f"torch_version={torch.__version__}")
        print(f"torch_cuda_version={getattr(torch.version, 'cuda', None)}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"cuda_device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch_probe_error={exc}")


if __name__ == "__main__":
    main()
