"""Generate the static paper figures that do not come out of the eval suite.

Writes into project_paper/LATEX/generated/:
  - plot_corpus_dates.png     two-panel publication-year histogram (624k | 13M)
  - plot_retrievability.png   distinct-docs-reached bar chart for the
                              retrievability probe (numbers parsed from the
                              suite-generated results_retrievability.tex)

Deterministic and offline-friendly: the 13M year counts are read live from
Postgres when reachable and otherwise fall back to the cached counts recorded
below (measured once against the loaded 13.36M instance). The 624k counts come
from the corpus-profile artifact written by scripts/29_profile_corpus.py.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = PROJECT_ROOT / "project_paper" / "LATEX" / "generated"
YEARLY_COUNTS_624K_CSV = PROJECT_ROOT / "outputs" / "corpus_profile" / "yearly_counts.csv"
RETRIEVABILITY_TEX = GENERATED_DIR / "results_retrievability.tex"
PROTOCOL_C_TEX = GENERATED_DIR / "results_protocol_c.tex"

# Worked-example plot artefact (c06 run on the 13M corpus); copied into
# generated/ so the paper build does not depend on outputs/ being present.
C06_PLOT_SRC = PROJECT_ROOT / "outputs" / "agent_runtime" / "agent_8a324dc15916" / "plots" / "framing_shift_over_time.png"
C06_PLOT_DST = GENERATED_DIR / "plot_c06_framing_shift.png"

PG_DSN = os.getenv("CORPUSAGENT2_PG_DSN", "postgresql://corpus:corpus@127.0.0.1:5432/corpus_db")
PG_TABLE = os.getenv("CORPUSAGENT2_PG_TABLE", "article_corpus")

# Fallback year counts for the 13.36M Geralt-Targaryen/CC-News instance,
# cached from a live GROUP BY over the loaded table on 2026-07-03
# (year-level dating; sums to 13,358,521, all rows dated).
FALLBACK_13M_YEAR_COUNTS: dict[str, int] = {
    "2017": 3_100_804,
    "2018": 2_373_107,
    "2019": 3_089_903,
    "2020": 2_398_299,
    "2021": 2_396_408,
}

# Fallback for the retrievability plot if the generated table is absent.
FALLBACK_RETRIEVABILITY = {"Lexical (BM25)": 2320, "Dense (E5)": 2393, "Hybrid (RRF)": 2396}
RETRIEVABILITY_CEILING = 2400

# Fallback for the Protocol C plot if the generated table is absent:
# (transformation, system) -> (jaccard, ci_low, ci_high)
FALLBACK_PROTOCOL_C = {
    ("Paraphrase", "Hybrid + rerank"): (0.349, 0.22, 0.49),
    ("Paraphrase", "Dense (E5)"): (0.485, 0.29, 0.66),
    ("Entity swap", "Hybrid + rerank"): (0.291, 0.14, 0.47),
    ("Entity swap", "Dense (E5)"): (0.293, 0.13, 0.48),
}


def load_624k_year_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(YEARLY_COUNTS_624K_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            year = str(row.get("year", "")).strip()
            label = "undated" if year in {"", "<missing>", "none"} else year
            counts[label] = counts.get(label, 0) + int(row["count"])
    return counts


def load_13m_year_counts() -> tuple[dict[str, int], str]:
    try:
        from psycopg import connect

        with connect(PG_DSN, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COALESCE(NULLIF(LEFT(published_at, 4), ''), 'undated') AS y, COUNT(*) "
                    f"FROM {PG_TABLE} GROUP BY 1 ORDER BY 1"
                )
                counts = {str(year): int(count) for year, count in cur.fetchall()}
        if counts:
            return counts, "live postgres"
    except Exception as exc:
        print(f"[figures] postgres unavailable ({exc}); using cached 13M counts")
    return dict(FALLBACK_13M_YEAR_COUNTS), "cached"


def parse_retrievability_counts() -> dict[str, int]:
    if not RETRIEVABILITY_TEX.exists():
        return dict(FALLBACK_RETRIEVABILITY)
    text = RETRIEVABILITY_TEX.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    for match in re.finditer(r"^(Lexical[^&]*|Dense[^&]*|Hybrid[^&]*)&[^&]*&\s*([\d,]+)\s*&", text, re.MULTILINE):
        label = match.group(1).replace("\\", "").strip()
        counts[label] = int(match.group(2).replace(",", ""))
    return counts or dict(FALLBACK_RETRIEVABILITY)


def sorted_year_items(counts: dict[str, int]) -> list[tuple[str, int]]:
    years = sorted((k, v) for k, v in counts.items() if k != "undated")
    if "undated" in counts:
        years.append(("undated", counts["undated"]))
    return years


def plot_corpus_dates(counts_624k: dict[str, int], counts_13m: dict[str, int], source_13m: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for ax, (counts, title) in zip(
        axes,
        [
            (counts_624k, "624k slice (vblagoje/cc_news)"),
            (counts_13m, "13.36M instance (Geralt-Targaryen/CC-News)"),
        ],
    ):
        items = sorted_year_items(counts)
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        colors = ["#888888" if k == "undated" else "#3b6ea5" for k in labels]
        ax.bar(labels, values, color=colors)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("documents", fontsize=8)
        ax.tick_params(labelsize=8)
        for idx, value in enumerate(values):
            ax.annotate(f"{value:,}", (idx, value), ha="center", va="bottom", fontsize=6.5)
        ax.margins(y=0.15)
    if not counts_13m:
        axes[1].text(0.5, 0.5, "13M counts unavailable", ha="center", va="center", transform=axes[1].transAxes)
    fig.suptitle("Publication-date distribution by year (grey = no timestamp)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = GENERATED_DIR / "plot_corpus_dates.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[figures] wrote {out} (13M source: {source_13m})")


def plot_retrievability(counts: dict[str, int]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    ax.bar(labels, values, color=["#b06a3b", "#3b6ea5", "#4a9a63"])
    ax.axhline(RETRIEVABILITY_CEILING, linestyle="--", color="black", linewidth=1)
    ax.annotate(
        f"ceiling = {RETRIEVABILITY_CEILING:,} slots (120 queries x top-20)",
        (0.02, RETRIEVABILITY_CEILING + RETRIEVABILITY_CEILING * 0.06),
        xycoords=("axes fraction", "data"),
        va="bottom",
        fontsize=8,
    )
    for idx, value in enumerate(values):
        ax.annotate(f"{value:,}", (idx, value * 0.5), ha="center", va="center", fontsize=9, color="white")
    ax.set_ylabel("distinct documents reached", fontsize=9)
    ax.set_ylim(0, RETRIEVABILITY_CEILING * 1.25)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    out = GENERATED_DIR / "plot_retrievability.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[figures] wrote {out}")


def parse_protocol_c() -> dict[tuple[str, str], tuple[float, float, float]]:
    if not PROTOCOL_C_TEX.exists():
        return dict(FALLBACK_PROTOCOL_C)
    text = PROTOCOL_C_TEX.read_text(encoding="utf-8")
    values: dict[tuple[str, str], tuple[float, float, float]] = {}
    row_re = re.compile(
        r"^(Paraphrase|Entity swap)[^&]*&\s*([^&]+?)\s*&\s*([\d.]+)\s*\{\\scriptsize\[([\d.]+),([\d.]+)\]\}",
        re.MULTILINE,
    )
    for m in row_re.finditer(text):
        values[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
    return values or dict(FALLBACK_PROTOCOL_C)


def plot_protocol_c(values: dict[tuple[str, str], tuple[float, float, float]]) -> None:
    systems = ["Dense (E5)", "Hybrid + rerank"]
    transforms = ["Paraphrase", "Entity swap"]
    colors = {"Paraphrase": "#3b6ea5", "Entity swap": "#b06a3b"}
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    width = 0.35
    for t_idx, transform in enumerate(transforms):
        xs, ys, errs = [], [], []
        for s_idx, system in enumerate(systems):
            val = values.get((transform, system))
            if val is None:
                continue
            j, lo, hi = val
            xs.append(s_idx + (t_idx - 0.5) * width)
            ys.append(j)
            errs.append([j - lo, hi - j])
        err_arr = list(zip(*errs)) if errs else None
        ax.bar(xs, ys, width=width, color=colors[transform],
               label=f"{transform} (expect {'HIGH' if transform == 'Paraphrase' else 'LOW'})",
               yerr=err_arr, capsize=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), ha="center", va="bottom", fontsize=8, xytext=(0, 10), textcoords="offset points")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel("top-25 Jaccard overlap", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = GENERATED_DIR / "plot_protocol_c.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[figures] wrote {out}")


def copy_c06_plot() -> None:
    if C06_PLOT_SRC.exists():
        C06_PLOT_DST.write_bytes(C06_PLOT_SRC.read_bytes())
        print(f"[figures] copied {C06_PLOT_SRC.name} -> {C06_PLOT_DST}")
    elif C06_PLOT_DST.exists():
        print(f"[figures] source run plot missing; keeping existing {C06_PLOT_DST.name}")
    else:
        print(f"[figures] WARNING: {C06_PLOT_SRC} missing and no cached copy present")


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    counts_624k = load_624k_year_counts()
    counts_13m, source_13m = load_13m_year_counts()
    plot_corpus_dates(counts_624k, counts_13m, source_13m)
    plot_retrievability(parse_retrievability_counts())
    plot_protocol_c(parse_protocol_c())
    copy_c06_plot()


if __name__ == "__main__":
    main()
