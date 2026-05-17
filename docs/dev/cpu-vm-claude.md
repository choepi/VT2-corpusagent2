# Running a second Claude Code instance on the CPU VM

The local Windows workstation runs CUDA. CorpusAgent2 also has to work on a
CPU-only VM (cloud deploy, professor demos). Fixing CPU-specific bugs by
round-tripping through `master` is slow because the local instance cannot
see VM-side test output directly.

The fix: run a second Claude Code session **on the VM itself**. Both
instances share the GitHub repo. They don't share memory or context — they
coordinate via git commits.

## One-time setup on the VM

```bash
# 1. Node.js >=18 (any flavour, nvm is easiest)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
exec $SHELL
nvm install --lts

# 2. Claude Code CLI
npm install -g @anthropic-ai/claude-code

# 3. Repo
git clone https://github.com/choepi/VT2-corpusagent2.git
cd VT2-corpusagent2

# 4. Python deps (CPU torch profile)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 5. Optional: copy memory dir from local so the VM-Claude already knows
#    your role and preferences. Skip if you want a clean slate.
#    On the Windows box:
#      scp -r C:\Users\Choep\.claude\projects\D--Programmieren-GitHub-VT2-corpusagent2\memory \
#          vm-user@vm-host:~/.claude/projects/<remote-project-id>/memory
#    (The remote project id is created the first time you run `claude` in
#    the repo directory on the VM — get it from `ls ~/.claude/projects/`.)
```

## Starting a session

Always run under `tmux` or `screen` so SSH disconnects don't kill the
session.

```bash
tmux new -s claude
cd ~/VT2-corpusagent2
claude          # first run prompts for login
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t claude`.

## Coordination protocol

The two instances cannot see each other in real time. Use git as the
exchange medium. Pick **one role per session** and stick to it.

### Suggested branch layout

```
master                              # protected, no direct work
feature/ui-overhaul-and-fallbacks   # current local CUDA work
feature/cpu-fixes                   # VM-side CPU-specific fixes
```

`feature/cpu-fixes` branches off whichever feature branch contains the
broken behaviour. When the VM is done, you merge it back into the feature
branch on the local box, not into master.

### Workflow

1. **Local (CUDA):** finish a chunk, commit, `git push`. Tell VM-Claude the
   branch name in your next prompt to it.
2. **VM (CPU):**
   - `git fetch && git checkout feature/cpu-fixes`
   - Reproduce, fix, run `pytest -q`, commit, push.
   - Write a one-line note in the commit body if the local instance needs
     to know something non-obvious (e.g. "skip this test on CUDA, model
     loads differently").
3. **Local:** `git pull --rebase`, verify CUDA path still works, run
   CUDA-only tests, push.

Rule of thumb: never have both instances editing the same file at the same
time. If you start editing locally, give VM-Claude a different file or
ask it to wait.

## Sharp edges

- **Per-machine memory.** Each instance has its own
  `~/.claude/projects/<id>/memory/`. User-profile memories (role, comm
  style) are portable; project memories you've built up on local apply to
  this codebase regardless of where Claude runs, so copy them over once
  and they're useful. Re-share preferences on the VM if you skip the copy.
- **No shared task list.** What local Claude is in the middle of doing is
  invisible to VM Claude. Brief each one at the start of its session.
- **Long CPU runs.** Don't have Claude stream a 30-minute test run — write
  logs to a file and have Claude `tail` it or read it after the fact:
  ```bash
  uv run pytest -q tests/test_agent_runtime.py > /tmp/run.log 2>&1 &
  ```
  Then `tail -n 200 /tmp/run.log` when needed.
- **Env vars.** `CORPUSAGENT2_DEVICE=cpu`, `CORPUSAGENT2_USE_OPENAI`, and
  the PG DSN are likely different on the VM. Keep a VM-specific `.env`;
  don't commit it.
- **Docker on the VM.** If the VM uses Docker compose, build with
  `CORPUSAGENT2_DOCKER_TORCH_PROFILE=cpu` (default). The Dockerfile
  already picks the right torch wheel.
- **Killing Claude mid-task.** `Ctrl-c` interrupts the current generation,
  not the underlying process. If a `pytest` Claude started is still
  running and you want to kill the session, list jobs with `jobs -l` and
  `kill %<n>` before exiting `tmux`.
- **Authentication on the VM.** Claude Code uses your account. If you log
  in there, that machine has a token. Treat the VM accordingly — don't
  log in on shared/throwaway hosts.

## When in doubt

- The repo's `CLAUDE.md` (root) is the source of truth for project
  conventions. Both instances read it on session start.
- If VM Claude and local Claude disagree on something, prefer the
  instance that observed the failure directly. The other is guessing.
