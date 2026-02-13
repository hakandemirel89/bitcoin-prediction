"""
PyTorch neural network for BTC 48h return prediction from IBIT flows.
Includes walk-forward training, signal generation, and tweet mode.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

class FlowMLP(nn.Module):
    """Simple feedforward MLP for tabular flow features → return prediction."""

    def __init__(self, n_features: int, hidden1: int = 64, hidden2: int = 32,
                 dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    max_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    device: str = "cpu",
) -> tuple:
    """
    Train FlowMLP on one fold.

    Returns: (model, train_losses, val_losses)
    """
    model = FlowMLP(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=patience)

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_v = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_losses = []
    val_losses = []

    n_samples = len(X_tr)

    for epoch in range(max_epochs):
        model.train()
        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()
        val_losses.append(val_loss)

        if early_stop.step(val_loss, model):
            logger.debug("Early stop at epoch %d", epoch + 1)
            break

    early_stop.restore_best(model)
    return model, train_losses, val_losses


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def walk_forward_train(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    min_train: int = 120,
    val_size: int = 30,
    test_size: int = 30,
    **train_kwargs,
) -> dict:
    """
    Walk-forward (expanding window) training and out-of-sample prediction.

    Returns dict with:
        - predictions: np.ndarray of OOS predictions (aligned with dates)
        - actuals: np.ndarray of actual values
        - pred_dates: pd.Series of dates for predictions
        - fold_results: list of per-fold info dicts
        - final_model: last trained model
        - final_scaler: last fitted scaler
    """
    n = len(X)
    n_features = X.shape[1]

    all_preds = []
    all_actuals = []
    all_dates = []
    fold_results = []

    fold = 0
    start = 0

    while start + min_train + val_size + test_size <= n:
        train_end = start + min_train + fold * test_size
        val_end = train_end + val_size
        test_end = min(val_end + test_size, n)

        if train_end >= n or val_end >= n:
            break

        # Split
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:test_end], y[val_end:test_end]
        test_dates = dates.iloc[val_end:test_end]

        if len(X_test) == 0:
            break

        # Scale (fit on train only — no leakage)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        # Train
        model, tr_loss, va_loss = train_model(
            X_train_s, y_train, X_val_s, y_val,
            n_features=n_features, **train_kwargs
        )

        # Predict on test
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test_s, dtype=torch.float32)
            preds = model(X_t).numpy()

        all_preds.append(preds)
        all_actuals.append(y_test)
        all_dates.append(test_dates.values)

        fold_results.append({
            "fold": fold,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "train_end_date": str(dates.iloc[train_end - 1]),
            "test_start_date": str(test_dates.iloc[0]),
            "test_end_date": str(test_dates.iloc[-1]),
            "best_val_loss": min(va_loss) if va_loss else float("inf"),
            "epochs_trained": len(tr_loss),
        })

        logger.info(
            "Fold %d: train=%d, val=%d, test=%d, best_val=%.6f",
            fold, len(X_train), len(X_val), len(X_test), fold_results[-1]["best_val_loss"]
        )

        fold += 1

        if test_end >= n:
            break

    predictions = np.concatenate(all_preds) if all_preds else np.array([])
    actuals = np.concatenate(all_actuals) if all_actuals else np.array([])
    pred_dates = pd.to_datetime(np.concatenate(all_dates)) if all_dates else pd.DatetimeIndex([])

    return {
        "predictions": predictions,
        "actuals": actuals,
        "pred_dates": pred_dates,
        "fold_results": fold_results,
        "final_model": model if fold > 0 else None,
        "final_scaler": scaler if fold > 0 else None,
    }


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------

def predictions_to_signals(
    predictions: np.ndarray,
    threshold: float = 0.005,
) -> np.ndarray:
    """
    Convert predicted log returns to trading signals.

    Returns: array of strings ('Long', 'Short', 'Flat')
    """
    signals = np.where(
        predictions > threshold, "Long",
        np.where(predictions < -threshold, "Short", "Flat")
    )
    return signals


# ---------------------------------------------------------------------------
# Save / Load Artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    model: nn.Module,
    scaler: StandardScaler,
    feature_cols: list,
    threshold: float,
    fold_results: list,
    directory: Optional[Path] = None,
) -> Path:
    """Save model, scaler, and metadata with timestamp."""
    save_dir = directory or ARTIFACTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model
    model_path = save_dir / f"model_{ts}.pt"
    torch.save(model.state_dict(), model_path)

    # Scaler
    scaler_path = save_dir / f"scaler_{ts}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Metadata
    meta = {
        "timestamp": ts,
        "feature_columns": feature_cols,
        "threshold": threshold,
        "n_features": len(feature_cols),
        "fold_results": fold_results,
    }
    meta_path = save_dir / f"metadata_{ts}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("Artifacts saved to %s with timestamp %s", save_dir, ts)
    return save_dir


def load_latest_artifacts(directory: Optional[Path] = None):
    """Load the most recent model, scaler, and metadata."""
    load_dir = directory or ARTIFACTS_DIR

    # Find latest metadata
    meta_files = sorted(load_dir.glob("metadata_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No artifacts found in {load_dir}")

    latest_meta = meta_files[-1]
    ts = latest_meta.stem.replace("metadata_", "")

    with open(latest_meta) as f:
        meta = json.load(f)

    # Load scaler
    with open(load_dir / f"scaler_{ts}.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load model
    n_features = meta["n_features"]
    model = FlowMLP(n_features)
    model.load_state_dict(torch.load(load_dir / f"model_{ts}.pt", weights_only=True))
    model.eval()

    return model, scaler, meta


# ---------------------------------------------------------------------------
# Tweet Mode
# ---------------------------------------------------------------------------

def generate_tweet(
    df: pd.DataFrame,
    model: nn.Module,
    scaler: StandardScaler,
    feature_cols: list,
    threshold: float = 0.005,
) -> str:
    """
    Generate a 'tweet mode' prediction text block from the latest data.

    Parameters:
        df: merged dataframe with features built
        model: trained PyTorch model
        scaler: fitted StandardScaler
        feature_cols: list of feature column names
        threshold: signal threshold
    """
    from .features import build_features

    feat_df = build_features(df)
    latest = feat_df.iloc[-1]

    # Get features for latest row
    x = latest[feature_cols].values.astype(np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x)

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_scaled, dtype=torch.float32)).item()

    # Signal
    if pred > threshold:
        signal = "LONG"
        emoji = "🟢"
    elif pred < -threshold:
        signal = "SHORT"
        emoji = "🔴"
    else:
        signal = "FLAT"
        emoji = "⚪"

    date_str = pd.Timestamp(latest["date"]).strftime("%Y-%m-%d")
    flow = latest.get("flow_usd", 0)
    btc_close = latest.get("btc_close", 0)
    pred_pct = pred * 100

    flow_sign = "+" if flow >= 0 else ""

    text = f"""BTC-IBIT Signal -- {date_str}
{'=' * 38}
Yesterday's IBIT Flow: {flow_sign}${flow:.1f}M
Latest BTC Close:      ${btc_close:,.0f}
Model Forecast (48h):  {pred_pct:+.2f}%
Signal:                {signal} {emoji}
{'=' * 38}
Educational only, not financial advice."""

    return text
