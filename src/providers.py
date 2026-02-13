"""
Data providers for IBIT flows and BTC prices.
Each provider tries online sources first, then falls back to local CSV.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# IBIT Flow Provider
# ---------------------------------------------------------------------------

def _parse_farside_cell(text: str) -> Optional[float]:
    """
    Parse a single Farside table cell value.
    Returns float (millions USD) or None for missing data.

    Formats:
      "111.7"      → 111.7   (positive)
      "(95.1)"     → -95.1   (negative, parenthesized)
      "0.0"        → 0.0     (zero)
      "-"          → None    (no data / holiday)
      ""           → None    (empty)
    """
    text = text.strip().replace(",", "").replace("$", "")
    if text in ("", "-", "–", "\u2013"):
        return None
    if text.startswith("(") and text.endswith(")"):
        try:
            return -float(text[1:-1])
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _fetch_ibit_flows_farside() -> Optional[pd.DataFrame]:
    """
    Scrape IBIT daily net flows from Farside Investors.

    The page has multiple <table> elements.  The data lives in
    <table class="etf"> which contains <thead> (header) and <tbody> (rows).
    Column values are wrapped in <span class="tabletext">;
    negatives additionally in <span class="redFont">(val)</span>.
    Summary rows (Total/Average/Max/Min) and the green "today" row
    are filtered out by date-parsing.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Target the data table specifically: <table class="etf">
        table = soup.find("table", class_="etf")
        if table is None:
            logger.warning("Farside: <table class='etf'> not found on page")
            return None

        # --- Parse header to find IBIT column index ---
        thead = table.find("thead")
        if thead is None:
            logger.warning("Farside: no <thead> in etf table")
            return None

        header_row = thead.find("tr")
        header_cells = header_row.find_all("th")
        header_texts = [c.get_text(strip=True) for c in header_cells]

        ibit_col = None
        for i, h in enumerate(header_texts):
            if h.upper() == "IBIT":
                ibit_col = i
                break
        if ibit_col is None:
            logger.warning("Farside: IBIT column not found in headers: %s", header_texts)
            return None

        logger.debug("Farside: IBIT is column %d of %d headers", ibit_col, len(header_texts))

        # --- Parse body rows ---
        tbody = table.find("tbody")
        if tbody is None:
            logger.warning("Farside: no <tbody> in etf table")
            return None

        records = []
        skipped = 0
        for row in tbody.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) <= ibit_col:
                continue

            # Date is in the first cell
            date_text = cells[0].get_text(strip=True)

            # Skip summary rows (Total, Average, Maximum, Minimum)
            if date_text.lower() in ("total", "average", "maximum", "minimum", ""):
                continue

            # Parse date — format is "DD Mon YYYY" e.g. "11 Jan 2024"
            try:
                date = pd.to_datetime(date_text, format="%d %b %Y")
            except (ValueError, TypeError):
                # If it doesn't parse as a date, skip (summary row, etc.)
                skipped += 1
                continue

            # Parse IBIT flow value
            flow_text = cells[ibit_col].get_text(strip=True)
            flow_val = _parse_farside_cell(flow_text)

            # None means no data (holiday/not yet reported) — skip
            if flow_val is None:
                continue

            records.append({"date": date, "flow_usd": flow_val})

        if skipped:
            logger.debug("Farside: skipped %d non-date rows", skipped)

        if not records:
            logger.warning("Farside: parsed 0 data rows")
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        logger.info("Farside: loaded %d rows of IBIT flow data (online)", len(df))
        return df

    except Exception as e:
        logger.warning("Farside scraping failed: %s", e)
        return None


def _load_ibit_flows_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Load IBIT flows from local CSV."""
    csv_path = path or DATA_DIR / "ibit_flows.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("CSV fallback: loaded %d rows from %s", len(df), csv_path)
    return df


def get_ibit_flows(csv_path: Optional[Path] = None, force_csv: bool = False) -> pd.DataFrame:
    """
    Get IBIT daily net flow data.
    Tries Farside Investors first, falls back to CSV.

    Returns DataFrame with columns: [date, flow_usd]
    flow_usd is in millions USD.
    """
    if not force_csv:
        df = _fetch_ibit_flows_farside()
        if df is not None and len(df) > 0:
            return df
        logger.info("Online IBIT source unavailable, falling back to CSV")

    return _load_ibit_flows_csv(csv_path)


# ---------------------------------------------------------------------------
# BTC Price Provider
# ---------------------------------------------------------------------------

def _fetch_btc_yfinance(start: str = "2024-01-01") -> Optional[pd.DataFrame]:
    """Fetch BTC daily OHLCV via yfinance."""
    try:
        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ticker = yf.Ticker("BTC-USD")
            hist = ticker.history(start=start, auto_adjust=True)

        if hist.empty:
            return None

        df = hist.reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date").reset_index(drop=True)
        logger.info("yfinance: loaded %d rows of BTC price data", len(df))
        return df

    except Exception as e:
        logger.warning("yfinance BTC fetch failed: %s", e)
        return None


def _fetch_btc_coingecko(start: str = "2024-01-01") -> Optional[pd.DataFrame]:
    """Fetch BTC daily prices from CoinGecko (free, no auth)."""
    try:
        import requests

        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp.now().timestamp())

        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": start_ts,
            "to": end_ts,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        prices = data.get("prices", [])
        if not prices:
            return None

        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.groupby("date").agg({"close": "last"}).reset_index()

        # CoinGecko market_chart doesn't give OHLV, approximate:
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df["close"] * (1 + np.random.uniform(0, 0.02, len(df)))
        df["low"] = df["close"] * (1 - np.random.uniform(0, 0.02, len(df)))
        df["volume"] = 0.0  # Not available from this endpoint

        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date").reset_index(drop=True)
        logger.info("CoinGecko: loaded %d rows of BTC price data", len(df))
        return df

    except Exception as e:
        logger.warning("CoinGecko BTC fetch failed: %s", e)
        return None


def _load_btc_prices_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Load BTC prices from local CSV."""
    csv_path = path or DATA_DIR / "btc_prices.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("CSV fallback: loaded %d rows from %s", len(df), csv_path)
    return df


def get_btc_prices(
    csv_path: Optional[Path] = None,
    start: str = "2024-01-01",
    force_csv: bool = False,
) -> pd.DataFrame:
    """
    Get BTC daily price data.
    Tries yfinance → CoinGecko → CSV fallback.

    Returns DataFrame with columns: [date, open, high, low, close, volume]
    """
    if not force_csv:
        # Try yfinance first
        df = _fetch_btc_yfinance(start)
        if df is not None and len(df) > 0:
            return df

        # Try CoinGecko
        df = _fetch_btc_coingecko(start)
        if df is not None and len(df) > 0:
            return df

        logger.info("All online BTC sources unavailable, falling back to CSV")

    return _load_btc_prices_csv(csv_path)


# ---------------------------------------------------------------------------
# Merge Function
# ---------------------------------------------------------------------------

def merge_data(
    ibit_df: pd.DataFrame,
    btc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge IBIT flows with BTC prices on date (inner join).
    Adds derived columns: flow_cum_30d, flow_cum_60d, btc_bought_sold.
    """
    # Normalize dates to date-only (no time component)
    ibit = ibit_df.copy()
    btc = btc_df.copy()
    ibit["date"] = pd.to_datetime(ibit["date"]).dt.normalize()
    btc["date"] = pd.to_datetime(btc["date"]).dt.normalize()

    # Inner join — only dates where both IBIT and BTC data exist
    merged = pd.merge(ibit, btc, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Rename for clarity
    if "close" in merged.columns and "btc_close" not in merged.columns:
        merged = merged.rename(columns={
            "open": "btc_open",
            "high": "btc_high",
            "low": "btc_low",
            "close": "btc_close",
            "volume": "btc_volume",
        })

    # Derived columns
    merged["flow_cum_30d"] = merged["flow_usd"].rolling(30, min_periods=1).sum()
    merged["flow_cum_60d"] = merged["flow_usd"].rolling(60, min_periods=1).sum()
    merged["btc_bought_sold"] = (merged["flow_usd"] * 1e6) / merged["btc_close"]

    return merged
