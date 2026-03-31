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

## 6. Docker (optional)

Requires [Docker](https://docs.docker.com/get-docker/) with BuildKit enabled (default on current Docker Desktop / Engine).

From **repository root** (or use `-f` paths as below):

```bash
docker build -f deploy/Dockerfile -t mft-cashcow-paper .
```

Run the continuous paper loop with **host-mounted** data dirs (same persistence as local) and secrets from **repo-root** `.env`:

```bash
cd deploy
docker compose up -d --build
```

- **`env_file`**: `deploy/docker-compose.yml` references `../.env`. Copy `.env.example` to `.env` and set keys, or point `env_file` at `/etc/mft-cashcow.env`.
- **Volumes**: `v3/data` and `v2/data` are bind-mounted so candles, DuckDB, and paper artifacts survive container restarts.
- **Logs**: `docker compose logs -f paper`

If `libta-lib0` is missing in your base image (unusual on `python:3.11-slim-bookworm`), adjust the `apt-get` line in `deploy/Dockerfile` per [TA-Lib install](https://github.com/ta-lib/ta-lib) for your distro.

Smoke test in a one-off container (from **repo root**, image already built):

```bash
docker run --rm --env-file .env \
  -v "$(pwd)/v3/data:/app/v3/data" -v "$(pwd)/v2/data:/app/v2/data" \
  mft-cashcow-paper python v3/scripts/smoke_paper_deploy.py --iterations 1
```

## 7. VPS provisioning (from scratch)

A **1 vCPU / 1–2 GB RAM** Linux VPS (Ubuntu 22.04+ or Debian 12+) is plenty for hourly bar logic. Providers like **Hetzner**, **DigitalOcean**, **Linode**, or **AWS Lightsail** all work — pick whatever is cheapest and closest to you.

### 7.1 Create the VPS

| Setting | Recommended |
|---------|-------------|
| OS | Ubuntu 24.04 LTS (or 22.04) |
| Plan | 1 vCPU, 1 GB RAM, 25 GB SSD (~$4–6/mo) |
| Region | Any — latency to Kraken is not critical at 1h bars |
| Auth | SSH key (add your `~/.ssh/id_ed25519.pub` at creation) |

After the VPS is live, note the **public IP**.

### 7.2 First SSH connection

```bash
# From your local machine
ssh root@<VPS_IP>
```

### 7.3 Automated bootstrap

The repo includes a setup script that handles everything:

```bash
# Option A: clone first, then run
git clone https://github.com/sunnycho100/MFT-cashcow.git /opt/mft-cashcow
bash /opt/mft-cashcow/deploy/scripts/vps-setup.sh

# Option B: run directly (script clones for you)
curl -sSL https://raw.githubusercontent.com/sunnycho100/MFT-cashcow/main/deploy/scripts/vps-setup.sh | bash
```

The script creates an `mft` deploy user, hardens SSH, sets up UFW, installs Python 3.11 + TA-Lib, creates the virtualenv, installs the systemd unit, and configures log rotation.

Customize with environment variables:

```bash
MFT_USER=trader MFT_INSTALL_DIR=/srv/cashcow MFT_SSH_PORT=2222 bash deploy/scripts/vps-setup.sh
```

### 7.4 Post-bootstrap checklist

```bash
# 1. Add your SSH key to the deploy user (if not done at VPS creation)
echo 'ssh-ed25519 AAAA...' >> /home/mft/.ssh/authorized_keys

# 2. Fill in API keys
sudo nano /etc/mft-cashcow.env

# 3. Smoke test
sudo -u mft bash -c '
  cd /opt/mft-cashcow
  source .venv/bin/activate
  set -a && source /etc/mft-cashcow.env && set +a
  python3 v3/scripts/smoke_paper_deploy.py
'

# 4. Enable the service (only after smoke test passes)
sudo systemctl enable --now mft-cashcow-paper.service

# 5. Watch logs
sudo journalctl -u mft-cashcow-paper.service -f
```

### 7.5 Updating the code on the VPS

```bash
sudo -u mft bash -c '
  cd /opt/mft-cashcow
  git pull --ff-only
  source .venv/bin/activate
  pip install -r v2/requirements.txt
'
sudo systemctl restart mft-cashcow-paper.service
```

### 7.6 Docker on the VPS (alternative to systemd)

If you prefer Docker over systemd on the VPS:

```bash
# Install Docker (official convenience script)
curl -fsSL https://get.docker.com | sh
usermod -aG docker mft

# Build and run
cd /opt/mft-cashcow
cp deploy/env.example .env
nano .env  # fill in keys
docker build -f deploy/Dockerfile -t mft-cashcow-paper .
cd deploy && docker compose up -d --build

# Logs
docker compose logs -f paper
```

### 7.7 Backups

Periodically pull artifacts off the VPS:

```bash
# From your local machine
rsync -avz mft@<VPS_IP>:/opt/mft-cashcow/v3/data/paper/ ./backups/paper/
rsync -avz mft@<VPS_IP>:/opt/mft-cashcow/v3/data/walkforward/ ./backups/walkforward/
```

Or set up a cron on the VPS to push to object storage (S3, Backblaze B2, etc.) if you prefer.
