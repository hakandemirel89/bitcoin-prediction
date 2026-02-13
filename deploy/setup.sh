#!/usr/bin/env bash
# =============================================================================
# Deployment setup for BTC-IBIT Prediction Dashboard
# Target: Raspberry Pi 5, Debian/Bookworm, https://lnodebtc.duckdns.org
#
# This script:
#   1. Installs Caddy (reverse proxy with auto-HTTPS)
#   2. Installs the systemd service for Streamlit
#   3. Configures Caddy with basic auth
#   4. Opens required firewall ports
#
# Usage:
#   chmod +x deploy/setup.sh
#   sudo deploy/setup.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="/home/pi/AI-Startup-Lab/bitcoin-prediction"
CADDY_CONFIG="/etc/caddy/Caddyfile"

echo "=========================================="
echo " BTC-IBIT Dashboard Deployment"
echo "=========================================="

# --- 1. Install Caddy ---
if ! command -v caddy &>/dev/null; then
    echo "[1/5] Installing Caddy..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
    sudo apt-get update -qq
    sudo apt-get install -y -qq caddy
else
    echo "[1/5] Caddy already installed: $(caddy version)"
fi

# --- 2. Set up basic auth ---
echo ""
echo "[2/5] Setting up basic auth..."
echo "  You need to set a password for the dashboard."
echo ""
read -sp "  Enter dashboard password: " DASH_PASS
echo ""

HASH=$(caddy hash-password --plaintext "$DASH_PASS")
echo "  Password hash generated."

# --- 3. Configure Caddy ---
echo "[3/5] Writing Caddy config..."
sudo mkdir -p /var/log/caddy
cat <<CADDYEOF | sudo tee "$CADDY_CONFIG" > /dev/null
lnodebtc.duckdns.org {
    handle /ibit* {
        basicauth {
            pi ${HASH}
        }

        reverse_proxy 127.0.0.1:8501

        header {
            X-Content-Type-Options nosniff
            X-Frame-Options DENY
            Referrer-Policy strict-origin-when-cross-origin
        }
    }

    log {
        output file /var/log/caddy/btc-predict.log
        format console
    }
}
CADDYEOF
echo "  Caddy config written to $CADDY_CONFIG"

# --- 4. Install Streamlit systemd service ---
echo "[4/5] Installing systemd service..."
sudo cp "${PROJECT_DIR}/deploy/btc-predict.service" /etc/systemd/system/btc-predict.service
sudo systemctl daemon-reload
sudo systemctl enable btc-predict.service
sudo systemctl restart btc-predict.service
echo "  btc-predict.service enabled and started."

# --- 5. Start/restart Caddy ---
echo "[5/5] Starting Caddy..."
sudo systemctl enable caddy
sudo systemctl restart caddy
echo "  Caddy started."

# --- Firewall (ufw if installed) ---
if command -v ufw &>/dev/null; then
    echo ""
    echo "Configuring firewall (ufw)..."
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    sudo ufw allow 22/tcp
    echo "  Ports 80, 443, 22 opened."
fi

echo ""
echo "=========================================="
echo " Deployment complete!"
echo "=========================================="
echo ""
echo " Dashboard URL:  https://lnodebtc.duckdns.org/ibit"
echo " Username:       pi"
echo " Password:       (the one you just entered)"
echo ""
echo " Services:"
echo "   sudo systemctl status btc-predict"
echo "   sudo systemctl status caddy"
echo ""
echo " Logs:"
echo "   journalctl -u btc-predict -f"
echo "   tail -f /var/log/caddy/btc-predict.log"
echo ""
echo " IMPORTANT: Ensure your router forwards ports 80 and 443"
echo " to this Raspberry Pi's local IP address."
echo "=========================================="
