#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# MFT-cashcow VPS bootstrap script
# Target: Ubuntu 22.04+ / Debian 12+ (fresh minimal install)
#
# Run as root (or with sudo) on a freshly provisioned VPS:
#   curl -sSL <raw-url>/deploy/scripts/vps-setup.sh | sudo bash
#
# Or locally after cloning:
#   sudo bash deploy/scripts/vps-setup.sh
#
# What it does:
#   1. Creates a non-root deploy user (mft)
#   2. Hardens SSH (key-only, no root login)
#   3. Sets up UFW firewall (SSH + outbound HTTPS only)
#   4. Installs Python 3.11, TA-Lib C library, build tools
#   5. Clones the repo (or copies from /tmp/mft-cashcow-upload)
#   6. Creates virtualenv and installs dependencies
#   7. Installs env template and systemd unit
#   8. Sets up log rotation
#
# After running, you still need to:
#   - Add your SSH public key to ~mft/.ssh/authorized_keys
#   - Fill in /etc/mft-cashcow.env with real API keys
#   - Run the smoke test
#   - Enable the systemd service
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
DEPLOY_USER="${MFT_USER:-mft}"
DEPLOY_GROUP="${MFT_GROUP:-mft}"
INSTALL_DIR="${MFT_INSTALL_DIR:-/opt/mft-cashcow}"
REPO_URL="${MFT_REPO_URL:-https://github.com/sunnycho100/MFT-cashcow.git}"
BRANCH="${MFT_BRANCH:-main}"
PYTHON_VERSION="3.11"
ENV_FILE="/etc/mft-cashcow.env"
SSH_PORT="${MFT_SSH_PORT:-22}"

# ── Helpers ──────────────────────────────────────────────────────
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()   { printf '\033[1;31m[ERR]\033[0m   %s\n' "$*" >&2; }

require_root() {
  if [[ $EUID -ne 0 ]]; then
    err "This script must be run as root (or with sudo)."
    exit 1
  fi
}

# ── 0. Preflight ─────────────────────────────────────────────────
require_root
info "Starting MFT-cashcow VPS bootstrap..."
info "Deploy user: $DEPLOY_USER | Install dir: $INSTALL_DIR"

# ── 1. System packages ──────────────────────────────────────────
info "Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

info "Installing base dependencies..."
apt-get install -y -qq --no-install-recommends \
  git \
  curl \
  wget \
  unzip \
  software-properties-common \
  build-essential \
  libta-lib0 \
  libta-lib-dev \
  ufw \
  logrotate \
  jq

# ── 2. Python 3.11 ──────────────────────────────────────────────
if ! command -v "python${PYTHON_VERSION}" &>/dev/null; then
  info "Installing Python ${PYTHON_VERSION}..."
  add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    "python${PYTHON_VERSION}" \
    "python${PYTHON_VERSION}-venv" \
    "python${PYTHON_VERSION}-dev"
else
  info "Python ${PYTHON_VERSION} already installed."
fi

# ── 3. Deploy user ───────────────────────────────────────────────
if ! id "$DEPLOY_USER" &>/dev/null; then
  info "Creating deploy user: $DEPLOY_USER"
  groupadd -f "$DEPLOY_GROUP"
  useradd -m -g "$DEPLOY_GROUP" -s /bin/bash "$DEPLOY_USER"
  # Prepare SSH dir for key-based login
  install -d -m 700 -o "$DEPLOY_USER" -g "$DEPLOY_GROUP" "/home/$DEPLOY_USER/.ssh"
  touch "/home/$DEPLOY_USER/.ssh/authorized_keys"
  chmod 600 "/home/$DEPLOY_USER/.ssh/authorized_keys"
  chown "$DEPLOY_USER:$DEPLOY_GROUP" "/home/$DEPLOY_USER/.ssh/authorized_keys"
  warn "Add your SSH public key to /home/$DEPLOY_USER/.ssh/authorized_keys"
else
  info "User $DEPLOY_USER already exists."
fi

# ── 4. Firewall (UFW) ───────────────────────────────────────────
info "Configuring UFW firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow "$SSH_PORT"/tcp comment 'SSH'
# Allow outbound HTTPS for exchange APIs (already covered by default allow outgoing)
ufw --force enable
ufw status verbose

# ── 5. Harden SSH ───────────────────────────────────────────────
info "Hardening SSH configuration..."
SSHD_CONFIG="/etc/ssh/sshd_config"
if [[ -f "$SSHD_CONFIG" ]]; then
  # Backup original
  cp "$SSHD_CONFIG" "${SSHD_CONFIG}.bak.$(date +%Y%m%d%H%M%S)"

  # Apply hardening (idempotent sed)
  sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' "$SSHD_CONFIG"
  sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' "$SSHD_CONFIG"
  sed -i 's/^#\?ChallengeResponseAuthentication.*/ChallengeResponseAuthentication no/' "$SSHD_CONFIG"
  sed -i 's/^#\?UsePAM.*/UsePAM no/' "$SSHD_CONFIG"

  # Ensure settings exist if not already present
  grep -q '^PermitRootLogin' "$SSHD_CONFIG" || echo 'PermitRootLogin no' >> "$SSHD_CONFIG"
  grep -q '^PasswordAuthentication' "$SSHD_CONFIG" || echo 'PasswordAuthentication no' >> "$SSHD_CONFIG"

  systemctl reload sshd 2>/dev/null || systemctl reload ssh 2>/dev/null || true
  info "SSH hardened: root login disabled, password auth disabled."
  warn "Make sure your SSH key is in /home/$DEPLOY_USER/.ssh/authorized_keys BEFORE logging out!"
fi

# ── 6. Clone or copy repo ───────────────────────────────────────
if [[ -d "$INSTALL_DIR/.git" ]]; then
  info "Repo already at $INSTALL_DIR — pulling latest..."
  cd "$INSTALL_DIR"
  sudo -u "$DEPLOY_USER" git pull --ff-only origin "$BRANCH" || warn "Pull failed; using existing checkout."
elif [[ -d /tmp/mft-cashcow-upload ]]; then
  info "Copying uploaded repo from /tmp/mft-cashcow-upload..."
  cp -a /tmp/mft-cashcow-upload "$INSTALL_DIR"
else
  info "Cloning $REPO_URL (branch: $BRANCH)..."
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$INSTALL_DIR"
fi
chown -R "$DEPLOY_USER:$DEPLOY_GROUP" "$INSTALL_DIR"

# ── 7. Virtualenv & dependencies ────────────────────────────────
info "Setting up Python virtualenv..."
cd "$INSTALL_DIR"
sudo -u "$DEPLOY_USER" "python${PYTHON_VERSION}" -m venv .venv
sudo -u "$DEPLOY_USER" .venv/bin/pip install --upgrade pip wheel setuptools
sudo -u "$DEPLOY_USER" .venv/bin/pip install --no-cache-dir -r v2/requirements.txt

# ── 8. Secrets template ─────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  info "Installing env template to $ENV_FILE"
  cp "$INSTALL_DIR/deploy/env.example" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  chown root:root "$ENV_FILE"
  warn "Edit $ENV_FILE and fill in KRAKEN_API_KEY / KRAKEN_API_SECRET"
else
  info "$ENV_FILE already exists — not overwriting."
fi

# ── 9. Data directories ─────────────────────────────────────────
info "Ensuring data directories exist..."
sudo -u "$DEPLOY_USER" mkdir -p \
  "$INSTALL_DIR/v3/data/paper" \
  "$INSTALL_DIR/v3/data/walkforward" \
  "$INSTALL_DIR/v3/data/coinbase" \
  "$INSTALL_DIR/v3/data/deribit" \
  "$INSTALL_DIR/v2/data"

# ── 10. systemd unit ────────────────────────────────────────────
info "Installing systemd service..."
cp "$INSTALL_DIR/deploy/systemd/mft-cashcow-paper.service" /etc/systemd/system/
systemctl daemon-reload
info "Service installed but NOT enabled. Enable after smoke test passes:"
info "  sudo systemctl enable --now mft-cashcow-paper.service"

# ── 11. Log rotation ────────────────────────────────────────────
info "Setting up log rotation..."
cat > /etc/logrotate.d/mft-cashcow <<'EOF'
/opt/mft-cashcow/v3/data/paper/*.jsonl
/opt/mft-cashcow/v3/data/paper/*.log {
    weekly
    rotate 12
    compress
    delaycompress
    missingok
    notifempty
    create 0644 mft mft
}
EOF

# ── 12. Summary ──────────────────────────────────────────────────
cat <<DONE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VPS bootstrap complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Install dir:  $INSTALL_DIR
  Deploy user:  $DEPLOY_USER
  Env file:     $ENV_FILE
  Python venv:  $INSTALL_DIR/.venv

  Next steps:
  ──────────
  1. Add your SSH public key:
     echo 'ssh-ed25519 AAAA...' >> /home/$DEPLOY_USER/.ssh/authorized_keys

  2. Fill in API keys:
     sudo nano $ENV_FILE

  3. Run smoke test:
     sudo -u $DEPLOY_USER bash -c '
       cd $INSTALL_DIR
       source .venv/bin/activate
       set -a && source $ENV_FILE && set +a
       python3 v3/scripts/smoke_paper_deploy.py
     '

  4. Enable the service:
     sudo systemctl enable --now mft-cashcow-paper.service

  5. Watch logs:
     sudo journalctl -u mft-cashcow-paper.service -f

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DONE
