# Bitcoin Prediction from IBIT ETF Flows

Predict BTC 48-hour price movement using BlackRock IBIT ETF daily net inflow/outflow data. Walk-forward neural network with full USD + BTC backtest.

See **`plan.md`** for full architecture, assumptions, and design decisions.

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo notebook
python notebooks/demo.py

# 4. Run tests
python -m pytest tests/ -v
```

## Project Structure

```
├── plan.md              # Architecture & design doc (single source of truth)
├── requirements.txt     # Dependencies
├── data/                # CSV fallback data
├── src/
│   ├── providers.py     # Data ingestion (online + CSV fallback)
│   ├── features.py      # Feature engineering (no leakage)
│   ├── model.py         # PyTorch MLP, walk-forward training
│   ├── backtest.py      # Backtest engine (USD + BTC accounting)
│   └── viz.py           # Plotly charts
├── notebooks/
│   └── demo.py          # Main notebook (cell markers for Jupyter)
├── artifacts/           # Saved models + scalers
└── tests/               # Unit tests
```

## Data Sources

- **IBIT Flows**: Farside Investors (online) → `data/ibit_flows.csv` (fallback)
- **BTC Prices**: yfinance → CoinGecko → `data/btc_prices.csv` (fallback)

## Key Design Decisions

- **Target**: 48h forward log return (2 trading days)
- **Short**: Inverse-tracking model (no leverage required)
- **Validation**: Walk-forward expanding window (no leakage)
- **Initial portfolio**: 1 BTC
- **Transaction cost**: 10 bps per position change
