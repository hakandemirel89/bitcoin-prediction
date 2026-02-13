# Bitcoin Prediction from IBIT Flows — Master Plan

## Changelog
| Date       | Change | Files Impacted |
|------------|--------|----------------|
| 2025-02-13 | Initial plan created (Opus Plan Mode) | `plan.md` |
| 2025-02-13 | Full implementation completed (Sonnet Implementation) | All files below |
| 2025-02-13 | PyTorch pinned to 2.6.0 (2.10.0 crashes on RPi5 aarch64 — SIGILL) | `requirements.txt` |
| 2025-02-13 | Walk-forward min_train reduced to 90 (limited data: 253 usable samples) | `notebooks/demo.py` |
| 2025-02-13 | Verified: 16/16 tests pass, full pipeline runs end-to-end on RPi5 | — |
| 2026-02-13 | Fixed Farside scraper: target `table.etf` (not first table), parse `thead`/`tbody`, handle `redFont` parens for negatives | `src/providers.py` |
| 2026-02-13 | Both online sources working: Farside (523 rows IBIT flows) + yfinance (774 rows BTC prices) | `src/providers.py` |
| 2026-02-13 | CSV fallbacks updated with real data from online sources | `data/ibit_flows.csv`, `data/btc_prices.csv` |
| 2026-02-13 | Demo default changed from `force_csv=True` to `force_csv=False` (online first) | `notebooks/demo.py` |
| 2026-02-13 | Added Streamlit dashboard (4 pages: Data Explorer, Model, Backtest, Daily Prediction) | `app.py` |
| 2026-02-13 | Added deployment: systemd service, Caddy reverse proxy, setup script | `deploy/*` |
| 2026-02-13 | Updated .gitignore (secrets, .env, streamlit, logs), requirements.txt (+streamlit) | `.gitignore`, `requirements.txt` |
| 2026-02-13 | Full README rewrite: dashboard usage, training, deployment on RPi, security | `README.md` |
| 2026-02-13 | Updated plan.md: new sections 16 (Dashboard), 17 (Deployment) | `plan.md` |

---

## 1. Project Overview

Predict BTC price movement over the next **48 hours** using BlackRock IBIT ETF daily net inflow/outflow data as the primary signal. Run a no-leverage backtest tracking both USD and BTC equity. All code is notebook-first Python, runnable on a Raspberry Pi 5 (aarch64, 8 GB RAM).

---

## 2. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │ IBIT Provider │    │ BTC Provider  │                      │
│  │ (online/CSV)  │    │ (online/CSV)  │                      │
│  └──────┬───────┘    └──────┬───────┘                       │
│         │                   │                                │
│         ▼                   ▼                                │
│  ┌─────────────────────────────────────┐                    │
│  │        Merged DataFrame             │                    │
│  │  date | flow_usd | btc_close | ...  │                    │
│  └──────────────┬──────────────────────┘                    │
│                 │                                            │
├─────────────────┼───────────────────────────────────────────┤
│                 ▼          FEATURE LAYER                     │
│  ┌─────────────────────────────────────┐                    │
│  │      Feature Engineering            │                    │
│  │  lags, rolling, cumulative, ratios  │                    │
│  └──────────────┬──────────────────────┘                    │
│                 │                                            │
├─────────────────┼───────────────────────────────────────────┤
│                 ▼          MODEL LAYER                       │
│  ┌─────────────────────────────────────┐                    │
│  │   PyTorch Neural Network            │                    │
│  │   Walk-forward training             │                    │
│  │   → artifacts/ (model + scaler)     │                    │
│  └──────────────┬──────────────────────┘                    │
│                 │                                            │
├─────────────────┼───────────────────────────────────────────┤
│                 ▼          BACKTEST LAYER                    │
│  ┌─────────────────────────────────────┐                    │
│  │   Backtest Engine                   │                    │
│  │   Long / Short / Flat               │                    │
│  │   USD + BTC equity tracking         │                    │
│  └──────────────┬──────────────────────┘                    │
│                 │                                            │
├─────────────────┼───────────────────────────────────────────┤
│                 ▼          OUTPUT LAYER                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐                │
│  │  Charts  │ │  Tables  │ │  Tweet Mode  │                │
│  │ (plotly) │ │ (pandas) │ │  (text block)│                │
│  └─────┬────┘ └─────┬────┘ └──────┬───────┘                │
│        │            │             │                          │
├────────┼────────────┼─────────────┼─────────────────────────┤
│        ▼            ▼             ▼      DASHBOARD LAYER     │
│  ┌─────────────────────────────────────────┐                │
│  │   Streamlit App (app.py)                │                │
│  │   Pages: Data / Model / Backtest / Pred │                │
│  └──────────────┬──────────────────────────┘                │
│                 │                                            │
├─────────────────┼───────────────────────────────────────────┤
│                 ▼          DEPLOYMENT LAYER                   │
│  ┌──────────┐ ┌────────────┐ ┌──────────────┐              │
│  │ systemd  │ │   Caddy    │ │ Let's Encrypt│              │
│  │ service  │ │ rev. proxy │ │  auto-HTTPS  │              │
│  └──────────┘ └────────────┘ └──────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. File Structure

```
bitcoin-prediction/
├── plan.md                     # This file — single source of truth
├── app.py                      # Streamlit dashboard (4 pages)
├── requirements.txt            # Pinned dependencies
├── README.md                   # Full usage + deployment docs
├── .gitignore
├── data/
│   ├── ibit_flows.csv          # Real IBIT flow data (Farside scraper)
│   └── btc_prices.csv          # Real BTC price data (yfinance)
├── src/
│   ├── __init__.py
│   ├── providers.py            # Data ingestion (online + CSV fallback)
│   ├── features.py             # Feature engineering
│   ├── model.py                # PyTorch model, training, inference
│   ├── backtest.py             # Backtest engine (USD + BTC accounting)
│   └── viz.py                  # Plotly charts
├── artifacts/                  # Saved models + scalers (timestamped)
├── notebooks/
│   └── demo.py                 # Notebook-style script (cell markers)
├── deploy/
│   ├── btc-predict.service     # systemd unit file
│   ├── Caddyfile               # Caddy reverse proxy template
│   └── setup.sh                # One-shot deployment script
└── tests/
    ├── test_features.py
    └── test_backtest.py
```

---

## 4. Data Providers (`src/providers.py`)

### 4.1 IBIT Flow Provider

- **Primary**: Attempt to scrape Farside Investors (`https://farside.co.uk/bitcoin-etf-flow-all-data/`) for daily IBIT net flows. Parse HTML table, extract IBIT column.
- **Fallback**: Load `data/ibit_flows.csv`
- **Schema**: `date (datetime64), flow_usd (float64)` — flow in millions USD
- **Notes**: Farside is the only free source for ETF-level flow data. If scraping fails (site changes, network error), fallback to CSV silently with a log warning.

### 4.2 BTC Price Provider

- **Primary**: `yfinance` — `BTC-USD` ticker, daily OHLCV.
- **Secondary fallback**: CoinGecko `/coins/bitcoin/market_chart` endpoint (no auth, 30 req/min).
- **Final fallback**: Load `data/btc_prices.csv`
- **Schema**: `date (datetime64), open, high, low, close, volume (all float64)`

### 4.3 Merged DataFrame

After loading both sources, merge on `date` (inner join on trading days where both IBIT and BTC data exist). IBIT only trades on US market days, so BTC data on weekends/holidays without IBIT data is dropped.

**Merged schema:**

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime64 | Trading date |
| `flow_usd` | float64 | IBIT net flow (millions USD) |
| `btc_close` | float64 | BTC daily close (USD) |
| `btc_open` | float64 | BTC daily open (USD) |
| `btc_high` | float64 | BTC daily high (USD) |
| `btc_low` | float64 | BTC daily low (USD) |
| `btc_volume` | float64 | BTC 24h volume |
| `flow_cum_30d` | float64 | Rolling 30-day cumulative flow |
| `flow_cum_60d` | float64 | Rolling 60-day cumulative flow |
| `btc_bought_sold` | float64 | `flow_usd * 1e6 / btc_close` |

---

## 5. Feature Engineering (`src/features.py`)

### 5.1 Features (all derived from IBIT flows + BTC close)

**Flow-based features:**
- `flow_lag_1` .. `flow_lag_7` — lagged daily flows (1–7 days)
- `flow_roll_sum_3`, `flow_roll_sum_7`, `flow_roll_sum_30` — rolling sum
- `flow_roll_mean_7`, `flow_roll_mean_30` — rolling mean
- `flow_roll_std_7`, `flow_roll_std_30` — rolling std
- `flow_cum_30d`, `flow_cum_60d` — cumulative 30/60 day sums
- `flow_zscore_30d` — `(flow - roll_mean_30) / roll_std_30`

**Price-derived features (secondary):**
- `btc_ret_1d` — 1-day log return of BTC
- `btc_ret_5d` — 5-day log return
- `btc_volatility_7d` — 7-day rolling std of log returns
- `btc_bought_sold` — flow_usd_actual / btc_close (daily BTC equivalent)

**Total features: ~20** — kept small for the Pi's limited resources.

### 5.2 Target Variable

**`target_48h_ret`** = `log(btc_close[t+2] / btc_close[t])`

- 48h ≈ 2 trading days forward (since IBIT is only available on trading days).
- Log return chosen because: (a) stationary, (b) additive, (c) symmetric for gains/losses.
- For signal generation: positive predicted return → Long, negative → Short, near-zero (within threshold) → Flat.

### 5.3 Leakage Protection

- All features use **only past data** (lags, backward-looking rolling windows).
- Target is computed from **future** prices and is NEVER included in features.
- Walk-forward validation ensures train set is always strictly before test set.
- Scaler is fit on training data only; transform applied to val/test.

---

## 6. Model Design (`src/model.py`)

### 6.1 Architecture

Simple feedforward neural network (MLP) — appropriate for tabular data on resource-constrained hardware.

```
Input (n_features) → Linear(64) → ReLU → Dropout(0.3)
                   → Linear(32) → ReLU → Dropout(0.2)
                   → Linear(1)  → Output (predicted log return)
```

- **Loss**: MSE (regression on log returns)
- **Optimizer**: Adam, lr=1e-3, weight_decay=1e-5
- **Batch size**: 32
- **Max epochs**: 200
- **Early stopping**: patience=15, monitor validation loss

### 6.2 Walk-Forward Training

```
Total data: [========================================]

Split 1: [TRAIN====][VAL][TEST]
Split 2: [TRAIN=========][VAL][TEST]
Split 3: [TRAIN==============][VAL][TEST]
...
Final:   [TRAIN========================][VAL][TEST]
```

- Minimum training window: 90 days (reduced from planned 120 due to limited data — 253 usable samples)
- Validation window: 20 days
- Test (out-of-sample prediction) window: 20 days
- Expanding window: each fold adds the previous test window to training
- Number of folds depends on data length; typically 5–10 folds for ~2 years of data.

### 6.3 Signal Generation

From the model's predicted 48h log return:
- `pred > +threshold` → **Long** (hold BTC)
- `pred < -threshold` → **Short** (short BTC)
- otherwise → **Flat** (hold USD)

Default threshold: **0.005** (0.5% predicted move). Configurable.

### 6.4 Artifacts

Saved to `artifacts/` with timestamp:
- `artifacts/model_YYYYMMDD_HHMMSS.pt` — PyTorch state dict
- `artifacts/scaler_YYYYMMDD_HHMMSS.pkl` — sklearn StandardScaler (pickle)
- `artifacts/metadata_YYYYMMDD_HHMMSS.json` — feature names, threshold, training dates, metrics

---

## 7. Backtest Engine (`src/backtest.py`)

### 7.1 Accounting Model

**Initial portfolio: 1 BTC** (starting USD value = `btc_close[day_0]`).

**Position definitions (no leverage):**

| Position | What You Hold | USD Value | BTC Value |
|----------|--------------|-----------|-----------|
| **Long** | 1 BTC | btc_close × qty_btc | qty_btc |
| **Flat** | USD cash | cash_usd | cash_usd / btc_close |
| **Short** | USD cash + short obligation | 2 × entry_price - current_price (per unit) | (2 × entry_price - current_price) / current_price |

**Short position accounting (no leverage, simplified):**

When entering short from long:
1. Sell 1 BTC at current price → receive `entry_price` USD
2. Notionally "borrow" 1 BTC and sell it → receive another `entry_price` USD
3. Total USD held: `2 × entry_price`
4. Obligation: return 1 BTC

**BUT** — this requires leverage (borrowing). Since the requirement is **no leverage**:

**Simplified short (cash-and-invert model):**
- Short = "go to cash, and profit from BTC decline"
- When going short: sell all BTC → hold USD
- USD PnL while short: if BTC drops 5%, your USD buys 5.26% more BTC when you re-enter
- Effectively: Short ≡ Flat (hold USD). The "short" signal means "expect BTC to drop, so hold USD."

**This is the only consistent no-leverage interpretation.** Documented here as a key assumption.

> **Assumption**: Without leverage, Short ≡ Flat ≡ hold USD. The signal still distinguishes Long vs Short/Flat for model evaluation, but the backtest PnL treats Short and Flat identically (both hold USD). If the user wants true short exposure, leverage would be required.

**UPDATE**: To make Short distinct from Flat and more interesting for the backtest, we use an **inverse-tracking** model:
- **Flat**: hold USD, no exposure.
- **Short**: track the *inverse* of BTC daily return on the notional. This simulates a 1x inverse BTC product (like an inverse ETF). The accounting:
  - `short_return_today = -1 × btc_daily_return`
  - `equity_usd *= (1 + short_return_today)`
- This is economically equivalent to a cash-settled short, which exists in practice (via futures, inverse ETFs, CFDs) without margin leverage.
- **Key**: this can lose more than 100% in theory if BTC doubles in one day (extremely unlikely for daily). We cap max loss at the position size.

### 7.2 Trade Execution

- Trades occur once per day at **daily close** price.
- Transaction cost: **10 bps** (0.10%) applied on each position change (Long↔Short, Long↔Flat, Short↔Flat).
- No slippage modeled (daily close assumed executable).

### 7.3 Equity Tracking

Two parallel equity series:
- **`equity_usd[t]`**: portfolio value in USD at each day's close.
- **`equity_btc[t]`**: portfolio value in BTC at each day's close = `equity_usd[t] / btc_close[t]`.

### 7.4 Metrics

| Metric | Definition |
|--------|-----------|
| **Total Return (USD)** | `(final_usd / initial_usd) - 1` |
| **Total Return (BTC)** | `(final_btc / initial_btc) - 1` |
| **Max Drawdown (USD)** | Max peak-to-trough decline in `equity_usd` |
| **Max Drawdown (BTC)** | Max peak-to-trough decline in `equity_btc` |
| **Hit Rate** | % of trades where exit PnL > 0 |
| **Sharpe Ratio** | `mean(daily_returns) / std(daily_returns) × sqrt(252)` (annualized, simple) |
| **# Trades** | Count of position changes |
| **Exposure %** | % of days with a non-flat position |
| **Buy & Hold Return** | BTC return over same period (benchmark) |

### 7.5 Outputs

- `backtest_equity_df`: DataFrame with `date, position, equity_usd, equity_btc, btc_close`
- `trades_df`: DataFrame with `date, signal, entry_price, exit_price, pnl_usd, pnl_btc`
- `metrics_dict`: Dictionary of all metrics above

---

## 8. Visualization (`src/viz.py`)

All charts use **Plotly** for interactive output in notebooks.

1. **IBIT Daily Flow** — bar chart (green/red for inflow/outflow)
2. **BTC Close** — line chart with flow overlay
3. **Equity Curve (USD)** — strategy vs buy-and-hold
4. **Equity Curve (BTC)** — strategy vs holding 1 BTC
5. **Flow Summary Table** — DataFrame display matching requested columns

---

## 9. Tweet Mode (`src/model.py` or notebook function)

```
📊 BTC-IBIT Signal — 2025-02-13
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Yesterday's IBIT Flow: +$320.5M
Latest BTC Close: $97,432
Model Forecast (48h): +1.23%
Signal: LONG ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Educational only, not financial advice.
```

Function: `generate_tweet(df, model, scaler, feature_cols)` → returns formatted string.

---

## 10. Notebook Workflow (`notebooks/demo.py`)

The notebook is structured with `# %%` cell markers (VS Code / Jupyter compatible).

**Cell order:**
1. **Setup** — imports, paths, seed
2. **Data Loading** — call providers, merge, display summary table
3. **Feature Engineering** — build features, show sample
4. **Model Training** — walk-forward, show train/val losses
5. **Backtest** — run engine, display metrics
6. **Charts** — all 4 charts
7. **Tweet Mode** — generate prediction
8. **Save Artifacts** — model + scaler + metadata

---

## 11. Implementation Plan (Ordered Steps)

| Step | File | Description |
|------|------|-------------|
| 1 | `requirements.txt` | Dependencies |
| 2 | `data/ibit_flows.csv` | Sample IBIT flow data (Jan 2024 – Feb 2025) |
| 3 | `data/btc_prices.csv` | Sample BTC price data (Jan 2024 – Feb 2025) |
| 4 | `src/__init__.py` | Empty init |
| 5 | `src/providers.py` | IBIT + BTC data providers with fallback |
| 6 | `src/features.py` | Feature engineering + target |
| 7 | `src/model.py` | PyTorch MLP, training loop, walk-forward, tweet mode |
| 8 | `src/backtest.py` | Backtest engine with USD+BTC accounting |
| 9 | `src/viz.py` | Plotly charts |
| 10 | `notebooks/demo.py` | Main notebook script |
| 11 | `tests/test_features.py` | Feature engineering tests |
| 12 | `tests/test_backtest.py` | Backtest logic tests |
| 13 | `README.md` | Quick-start guide |
| 14 | Update `plan.md` | Reflect final implementation state |

---

## 12. Key Assumptions

1. **48h ≈ 2 trading days** (IBIT trades M–F only; BTC trades 24/7 but we align to IBIT calendar).
2. **Short = inverse-tracking** (daily inverse return, no leverage, capped at position size).
3. **Transaction cost = 10 bps** per position change.
4. **Initial portfolio = 1 BTC**.
5. **Trade at daily close** after IBIT data is available.
6. **No data leakage**: features are backward-looking; scaler fit on train only; walk-forward splits.
7. **Flow data in millions USD** — multiplied by 1e6 for actual USD values in BTC-bought calculation.

---

## 13. Known Limitations

- **IBIT flow data availability**: Farside scraping is brittle; CSV fallback is primary reliable path.
- **Short accounting**: Inverse-tracking model is a simplification. Real shorting involves margin, borrowing costs, and liquidation risk.
- **Raspberry Pi constraints**: Model is kept small (MLP with 64/32 hidden). Training may take a few minutes per fold.
- **Feature set**: Primarily flow-based. Could be enhanced with on-chain data, sentiment, macro indicators.
- **Daily granularity**: 48h prediction on daily bars means only 2 bars forward; intraday data could improve granularity.
- **Sample size**: ~1 year of IBIT data (launched Jan 2024) is limited for ML.

---

## 14. Future Improvements

- LSTM/Transformer architecture for sequence modeling
- Additional data sources (on-chain, sentiment, macro)
- Intraday IBIT flow estimates
- Automated daily data refresh + prediction pipeline (cron)
- Ensemble models with confidence calibration
- Dashboard: scheduled model retraining button with progress

---

## 15. Streamlit Dashboard (`app.py`)

Single-file Streamlit app with 4 pages. Reuses all `src/` modules — zero logic duplication.

### 15.1 Pages

| Page | Purpose | Key Controls |
|------|---------|-------------|
| **Data Explorer** | IBIT flow table, BTC charts, flow overlay | Row count slider, force-CSV toggle |
| **Model** | View/train model, artifact info, fold results | Train params (min_train, val, test, lr, epochs, patience, threshold) |
| **Backtest** | Interactive backtest with equity curves, drawdown, trades | Date range, threshold, tx cost bps, base asset (USD/BTC) |
| **Daily Prediction** | Latest signal, tweet-ready text block | — (uses latest artifact) |

### 15.2 Data Caching

- `@st.cache_data(ttl=3600)` for data loading and feature building.
- Force-CSV toggle in sidebar for offline mode.
- Training triggers `st.rerun()` after artifact save.

### 15.3 Backtest Page Controls

The Backtest page runs the backtest engine in real-time with user-adjustable:
- **Signal threshold**: 0.001–0.020 (slider)
- **Transaction cost**: 0–50 bps (slider)
- **Date range**: date pickers constrained to data range
- **Base asset**: USD or BTC (for drawdown chart axis)

Note: The backtest page uses the latest saved model/scaler to generate predictions over the full feature dataset, then filters by the user's date range. This is NOT walk-forward — it uses a single model for all predictions. Walk-forward OOS predictions are only available via the CLI notebook.

---

## 16. Deployment (`deploy/`)

### 16.1 Architecture

```
Internet → Router (port 80/443) → Caddy (auto-HTTPS + basic auth)
         → 127.0.0.1:8501 (Streamlit, systemd-managed)
```

### 16.2 Components

| Component | File | Purpose |
|-----------|------|---------|
| systemd service | `deploy/btc-predict.service` | Runs Streamlit on boot, auto-restarts |
| Caddy config | `deploy/Caddyfile` | Reverse proxy, auto-HTTPS, basic auth, security headers |
| Setup script | `deploy/setup.sh` | One-shot install: Caddy + systemd + password prompt |

### 16.3 Security

- **Basic auth**: username `admin`, bcrypt-hashed password (set during setup)
- **HTTPS**: Automatic via Let's Encrypt (Caddy handles cert provisioning + renewal)
- **Streamlit binding**: `127.0.0.1` only (never exposed directly)
- **Security headers**: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`
- **Firewall**: Only ports 22, 80, 443 open

### 16.4 Domain

Target domain: `lnodebtc.duckdns.org`
Requires: router port forwarding 80+443 → Pi local IP, DuckDNS pointing to public IP.

### 16.5 Checklist

- [ ] DuckDNS domain `lnodebtc` configured and pointing to Pi's public IP
- [ ] Router forwards port 80 (HTTP) → Pi's local IP
- [ ] Router forwards port 443 (HTTPS) → Pi's local IP
- [ ] `sudo deploy/setup.sh` executed (installs Caddy, sets password, starts services)
- [ ] `sudo systemctl status btc-predict` shows active
- [ ] `sudo systemctl status caddy` shows active
- [ ] `https://lnodebtc.duckdns.org` loads with basic auth prompt

---

## 17. Implementation Status (Final)

**All deliverables completed and verified.**

### Run Results — Real Data (2026-02-13, RPi5)
- **Data**: 523 IBIT flow rows (Farside online) + 774 BTC price rows (yfinance), merged to 523 rows
- **Usable samples**: 491 (after 30-day feature NaN drop)
- **Features**: 21 (7 lags, 6 rolling, 2 cumulative, 1 z-score, 3 price-derived, 1 btc-bought, 1 volatility)
- **Walk-forward**: 20 folds, 381 OOS predictions
- **Direction accuracy**: 52.0%
- **Correlation (pred vs actual)**: 0.054
- **Backtest**: -14.6% USD return vs +14.2% buy-and-hold, Sharpe -0.015
- **Signal mix**: 38% Long, 39% Short, 23% Flat
- **Tweet prediction (2026-02-11)**: -0.43% → FLAT

### Farside Scraper Details
- Target: `<table class="etf">` (not first `<table>` which is site nav)
- Header in `<thead>`, data in `<tbody>`
- Negatives: `<span class="redFont">(95.1)</span>` → -95.1
- Holidays/missing: dash `"-"` → skipped
- Summary rows (Total/Average/Max/Min) filtered by date parse failure
- Requires `User-Agent` header (403 without)

### Platform Notes
- **PyTorch 2.10.0** produces SIGILL on Raspberry Pi 5 (Cortex-A76, missing SVE instructions). **Pinned to 2.6.0** which works.
- Training runs in ~2 min total for 20 folds on CPU.
- Plotly `.show()` attempts to open browser; in headless mode, use `fig.write_html()` or Jupyter.

### Dashboard
- `app.py`: 4-page Streamlit dashboard, verified starts cleanly on RPi5
- Reuses all `src/` modules, zero logic duplication
- Cached data loading with 1-hour TTL
- Force-CSV toggle for offline operation

### Deployment
- `deploy/btc-predict.service`: systemd unit, binds Streamlit to 127.0.0.1:8501
- `deploy/Caddyfile`: Caddy reverse proxy template with basic auth + auto-HTTPS
- `deploy/setup.sh`: interactive one-shot deploy script

### Tests
- `tests/test_features.py`: 7 tests — leakage checks, column validation, shape checks
- `tests/test_backtest.py`: 9 tests — long/short/flat accounting, tx costs, equity consistency, metrics
