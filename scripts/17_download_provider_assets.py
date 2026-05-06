from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _run(command: list[str]) -> None:
    print(f"[run] {' '.join(command)}")
    subprocess.run(command, check=True)


def _download_spacy_model(python_exe: str) -> None:
    try:
        import spacy
    except Exception as exc:
        print(f"[skip] spacy unavailable: {exc}")
        return
    try:
        spacy.load("en_core_web_sm")
        print("[ready] spacy model en_core_web_sm already available.")
        return
    except OSError:
        pass
    _run([python_exe, "-m", "spacy", "download", "en_core_web_sm"])


def _download_stanza_models() -> None:
    try:
        import stanza
    except Exception as exc:
        print(f"[skip] stanza unavailable: {exc}")
        return
    languages = os.getenv("CORPUSAGENT2_STANZA_LANGS", "en,de,fr,it")
    for lang in [item.strip() for item in languages.split(",") if item.strip()]:
        print(f"[run] stanza.download({lang})")
        stanza.download(lang, processors="tokenize,mwt,pos,lemma,depparse,ner")


def _download_nltk_assets(python_exe: str) -> None:
    try:
        import nltk  # noqa: F401
    except Exception as exc:
        print(f"[skip] nltk unavailable: {exc}")
        return
    packages = [
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "wordnet",
        "omw-1.4",
        "stopwords",
    ]
    _run([python_exe, "-m", "nltk.downloader", *packages])


def _download_textblob_assets() -> None:
    try:
        from textblob import download_corpora
    except Exception as exc:
        print(f"[skip] textblob unavailable: {exc}")
        return
    print("[run] textblob.download_corpora.download_all()")
    download_corpora.download_all()


def main() -> None:
    python_exe = sys.executable
    project_root = Path(__file__).resolve().parents[1]
    print(f"[info] project_root={project_root}")
    print(f"[info] python={python_exe}")
    _download_spacy_model(python_exe)
    _download_stanza_models()
    _download_nltk_assets(python_exe)
    _download_textblob_assets()
    print("[done] provider assets downloaded.")


if __name__ == "__main__":
    main()
