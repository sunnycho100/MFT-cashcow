# Server deploy (v3 paper loop)

Artifacts here support an **always-on** host running the same loop as locally: [`v3/scripts/run_paper_loop.py`](../v3/scripts/run_paper_loop.py), with config in [`v3/config.yaml`](../v3/config.yaml).

See **[docs/server-host-plan.md](../docs/server-host-plan.md)** for strategy pinning, checklist, and risk notes.

## 1. Checkout and virtualenv

From the repo root (example paths use `/opt/mft-cashcow`; adjust as needed):

```bash
cd /opt/mft-cashcow
python3 -m venv .venv
source .venv/bin/activate
pip install -r v2/requirements.txt
```

Install any extra packages your environment needs for v3 (e.g. `loguru`, `pyyaml` if not already pulled in).

## 2. Secrets

```bash
sudo cp deploy/env.example /etc/mft-cashcow.env
sudo chmod 600 /etc/mft-cashcow.env
# Edit: set KRAKEN_API_KEY and KRAKEN_API_SECRET
```

Variable names must match `kraken.api_key_env` and `kraken.api_secret_env` in `v3/config.yaml`.

For **local development**, you can keep the same names in a repo-root **`.env`** file (see `.env.example`). It is gitignored and is loaded automatically when v3 code calls `load_config()` (requires `python-dotenv`).

## 3. Smoke test (prove the path before systemd)

From **repo root**, with the same env as production:

```bash
cd /opt/mft-cashcow
source .venv/bin/activate
set -a && source /etc/mft-cashcow.env && set +a
python3 v3/scripts/smoke_paper_deploy.py
```

This checks that `KRAKEN_*` (or whatever names you set in `v3/config.yaml`) are non-empty, runs **three** successful `IntegratedPaperRuntime.run_once()` cycles (fails fast on error), then confirms `paper.cycle_log_path` and `paper.artifact_path` exist and the cycle log is non-empty.

Options:

- `--iterations 1` for a quicker check
- `--skip-env-check` only for local debugging without keys (will likely fail on API calls)

Review output and `journalctl` only after this passes; then enable the systemd unit.

## 4. systemd

```bash
sudo cp deploy/systemd/mft-cashcow-paper.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mft-cashcow-paper.service
sudo journalctl -u mft-cashcow-paper.service -f
```

Edit the unit file **before** enabling: `User`, `WorkingDirectory`, `ExecStart` (Python path), and `EnvironmentFile` must match your server.

## 5. Manual run (no systemd)

```bash
cd /opt/mft-cashcow
source .venv/bin/activate
set -a && source /etc/mft-cashcow.env && set +a
python3 v3/scripts/run_paper_loop.py --iterations 0
```

Use `--iterations N` for a fixed number of cycles when testing.
