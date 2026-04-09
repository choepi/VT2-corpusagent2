from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys


TUNNEL_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a Cloudflare Quick Tunnel for the local CorpusAgent2 backend."
    )
    parser.add_argument(
        "--backend-url",
        default="http://127.0.0.1:8001",
        help="Local backend URL to expose.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        raise RuntimeError(
            "cloudflared is not installed. Install it first, then rerun this helper."
        )

    command = [cloudflared, "tunnel", "--url", args.backend_url]
    print(f"[run] {' '.join(command)}")
    print("[info] Keep this process running while you use the public demo URL.")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    announced_url = ""
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        if not announced_url:
            match = TUNNEL_URL_PATTERN.search(line)
            if match:
                announced_url = match.group(0)
                print("")
                print(f"[public-url] {announced_url}")
                print(
                    "[next] Use that HTTPS URL as the API Base URL in the GitHub Pages frontend."
                )
                print("")
    raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
