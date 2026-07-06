# legacy/

Root-level clutter moved here for overview. Nothing in this folder is needed
to run CorpusAgent2 (see repo README / CLAUDE.md for the live entry points).

| Entry | What it is / why it's here |
| --- | --- |
| `VT2-corpusagent2/` | Accidentally committed snapshot of a Postgres `pgdata/` directory (~40 MB, 1288 files). Candidate for deletion. |
| `git` | Stray empty file (likely a typo'd shell redirect). Candidate for deletion. |
| `main.py` | Pre-agent-runtime standalone retrieval probe. Superseded by `scripts/12_run_agent_api.py` / the V1 runtime. Referenced only in `docs/legacy/`. |
| `download_e5.py` | One-off helper to fetch the E5 model on the CUDA box. Model provisioning now goes through the HF cache volume / `deploy/bake_dense_model.py`. |
| `pyproject-Beast.toml` | Machine-specific pyproject variant for the "Beast" CUDA workstation. |
| `requirements-cu118.txt` | Generated export artifact of `scripts/19_rebuild_env_cuda118.sh` (the script regenerates it at repo root when rerun). |
| `web_old_apple/` | Previous frontend skin, replaced by `web/`. |
| `stray-corpusagent2-dir/` | Untracked stray `corpusagent2/models/e5-base-v2` dir created by a script run with a wrong relative path. |

Kept at root on purpose: `slurm/` (referenced by the active 13M cluster
workflow in `docs/repo_workflow.md` / `docs/script_relationships.md`), `log/`
(runtime log scaffolding with `.gitkeep`s), `data/` + `models/` + `outputs/`
(runtime state), `graphify-out/` (local tooling cache).
