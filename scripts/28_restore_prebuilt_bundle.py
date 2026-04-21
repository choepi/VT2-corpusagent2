from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_extract(archive: zipfile.ZipFile, target_root: Path) -> int:
    extracted = 0
    target_root = target_root.resolve()
    for member in archive.infolist():
        name = member.filename
        if not name or name.endswith("/"):
            continue
        destination = (target_root / name).resolve()
        destination.relative_to(target_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member, "r") as src, destination.open("wb") as dst:
            dst.write(src.read())
        extracted += 1
    return extracted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely restore a portable CorpusAgent2 prebuilt-data bundle into the repo root."
    )
    parser.add_argument("bundle_path", help="Path to the zip produced by scripts/27_build_prebuilt_bundle.py.")
    parser.add_argument("--target-root", default=str(REPO_ROOT), help="Repo root to restore into.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    bundle_path = Path(args.bundle_path).expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    target_root = Path(args.target_root).expanduser().resolve()
    with zipfile.ZipFile(bundle_path, "r") as archive:
        extracted = _safe_extract(archive, target_root)
    print(f"Restored {extracted} files into {target_root}")
    print("Next step: python scripts/22_prepare_vm_stack.py --skip-provider-assets")


if __name__ == "__main__":
    main()
