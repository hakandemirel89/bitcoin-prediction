# Bitcoin Prediction from IBIT ETF Flows

Predict BTC 48-hour price movement using BlackRock IBIT ETF daily net inflow/outflow data.
Walk-forward neural network with full USD + BTC backtest and a Streamlit dashboard.

See **`plan.md`** for full architecture, assumptions, and design decisions.

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo notebook (CLI)
python notebooks/demo.py

# 4. Run the Streamlit dashboard
streamlit run app.py

# 5. Run tests
python -m pytest tests/ -v
```

## Project Structure

```
bitcoin-prediction/
├── plan.md              # Architecture & design (single source of truth)
├── app.py               # Streamlit dashboard (4 pages)
├── requirements.txt     # Dependencies
├── data/                # CSV fallback data
│   ├── ibit_flows.csv   # Real IBIT flow data (Farside)
│   └── btc_prices.csv   # Real BTC price data (yfinance)
├── src/
│   ├── providers.py     # Data ingestion (online + CSV fallback)
│   ├── features.py      # Feature engineering (21 features, no leakage)
│   ├── model.py         # PyTorch MLP, walk-forward training
│   ├── backtest.py      # Backtest engine (USD + BTC accounting)
│   └── viz.py           # Plotly charts
├── notebooks/
│   └── demo.py          # CLI notebook (cell markers for Jupyter)
├── artifacts/           # Saved models + scalers (timestamped)
├── deploy/
│   ├── btc-predict.service  # systemd unit file
│   ├── Caddyfile            # Caddy reverse proxy config
│   └── setup.sh             # One-shot deployment script
└── tests/               # Unit tests (16 tests)
```

## Data Sources

| Source | Primary | Fallback |
|--------|---------|----------|
| IBIT Flows | Farside Investors (scraper) | `data/ibit_flows.csv` |
| BTC Prices | yfinance (`BTC-USD`) | `data/btc_prices.csv` |

Both online sources are optional. The system works fully offline with CSV fallback.

## Training a Model

```bash
source .venv/bin/activate

# Option A: via CLI notebook
python notebooks/demo.py

# Option B: via dashboard
streamlit run app.py
# → Navigate to "Model" page → click "Train Walk-Forward Model"
```

Artifacts are saved to `artifacts/` with timestamps:
- `model_YYYYMMDD_HHMMSS.pt` — PyTorch weights
- `scaler_YYYYMMDD_HHMMSS.pkl` — StandardScaler
- `metadata_YYYYMMDD_HHMMSS.json` — feature list, fold results

## Running the Backtest

Via the dashboard **Backtest** page, you can adjust:
- Signal threshold (default 0.005)
- Transaction cost (default 10 bps)
- Date range
- Base asset (USD or BTC)

Or via CLI: `python notebooks/demo.py` (runs full walk-forward + backtest).

## Running the Dashboard

```bash
source .venv/bin/activate
streamlit run app.py --server.port=8501
```

Dashboard pages:
1. **Data Explorer** — IBIT flow table, BTC price charts, flow overlay
2. **Model** — training controls, artifact info, fold results
3. **Backtest** — interactive backtest with equity curves, drawdown, trades
4. **Daily Prediction** — latest signal, tweet-ready text block

## Deployment on Raspberry Pi

The dashboard is designed to run on a Raspberry Pi 5 behind Caddy reverse proxy
with automatic HTTPS and basic auth.

### Prerequisites

- Raspberry Pi with Debian/Bookworm
- Domain pointing to your Pi's public IP (e.g., `lnodebtc.duckdns.org`)
- Router port forwarding: ports **80** and **443** → Pi's local IP

### One-Shot Deploy

```bash
# Run the setup script (installs Caddy, configures systemd + basic auth)
chmod +x deploy/setup.sh
sudo deploy/setup.sh
```

This will:
1. Install Caddy (auto-HTTPS via Let's Encrypt)
2. Prompt for a dashboard password (basic auth)
3. Install + enable the `btc-predict` systemd service
4. Start everything

### Manual Deploy

```bash
# 1. Install Caddy
sudo apt install caddy

# 2. Copy systemd service
sudo cp deploy/btc-predict.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now btc-predict

# 3. Generate password hash
caddy hash-password --plaintext 'YourPassword'

# 4. Edit /etc/caddy/Caddyfile (use deploy/Caddyfile as template)
#    Replace <HASH> with the hash from step 3
sudo systemctl restart caddy

# 5. Open firewall ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

### Service Management

```bash
# Dashboard service
sudo systemctl status btc-predict
sudo systemctl restart btc-predict
journalctl -u btc-predict -f

# Caddy (reverse proxy)
sudo systemctl status caddy
sudo systemctl restart caddy
tail -f /var/log/caddy/btc-predict.log
```

### Security Notes

- Dashboard is behind **basic auth** (username: `admin`)
- HTTPS is automatic via Let's Encrypt (Caddy handles renewal)
- Streamlit binds to `127.0.0.1` only (not directly exposed)
- Security headers: `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`
- Do **not** expose port 8501 directly; always go through Caddy

## Key Design Decisions

- **Target**: 48h forward log return (2 trading days)
- **Short**: Inverse-tracking model (no leverage required)
- **Validation**: Walk-forward expanding window (no data leakage)
- **Initial portfolio**: 1 BTC
- **Transaction cost**: 10 bps per position change

## PyTorch on Raspberry Pi

PyTorch 2.10.0 crashes with SIGILL on RPi5 (Cortex-A76). Use **2.6.0**:
```bash
pip install "torch>=2.6.0,<2.7"
```
