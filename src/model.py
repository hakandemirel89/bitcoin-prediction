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
    train_size: int = 120,
    val_size: int = 20,
    step_size: int = 1,
    progress_callback=None,
    **train_kwargs,
) -> dict:
    """
    Walk-forward (rolling window) training and out-of-sample prediction.

    For each prediction step, trains a fresh model on the previous
    `train_size` days, validates on the next `val_size` days, then
    predicts `step_size` day(s).  This produces one model per step —
    a truly rolling out-of-sample backtest with no look-ahead bias.

    Returns dict with:
        - predictions: np.ndarray of OOS predictions
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

    first_test = train_size + val_size
    total_steps = max(0, (n - first_test) // step_size)

    fold = 0
    cursor = first_test

    while cursor + step_size <= n:
        train_start = cursor - val_size - train_size
        train_end = cursor - val_size
        val_end = cursor
        test_end = min(cursor + step_size, n)

        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:test_end]
        y_test = y[val_end:test_end]
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
            "test_date": str(test_dates.iloc[0]),
            "best_val_loss": min(va_loss) if va_loss else float("inf"),
            "epochs_trained": len(tr_loss),
        })

        if fold % 50 == 0:
            logger.info(
                "Fold %d/%d: train=[%s..%s], test=%s, val_loss=%.6f",
                fold, total_steps,
                str(dates.iloc[train_start])[:10],
                str(dates.iloc[train_end - 1])[:10],
                str(test_dates.iloc[0])[:10],
                fold_results[-1]["best_val_loss"],
            )

        if progress_callback:
            progress_callback(fold + 1, total_steps)

        fold += 1
        cursor += step_size

    predictions = np.concatenate(all_preds) if all_preds else np.array([])
    actuals = np.concatenate(all_actuals) if all_actuals else np.array([])
    pred_dates = pd.to_datetime(np.concatenate(all_dates)) if all_dates else pd.DatetimeIndex([])

    logger.info("Walk-forward complete: %d folds, %d OOS predictions", fold, len(predictions))

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
    predictions_data: Optional[dict] = None,
    directory: Optional[Path] = None,
) -> Path:
    """Save model, scaler, metadata, and optionally OOS predictions."""
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

    # Sample fold results for metadata (keep every 10th for display)
    sampled_folds = fold_results[::max(1, len(fold_results) // 20)] if fold_results else []
    avg_val_loss = float(np.mean([f["best_val_loss"] for f in fold_results])) if fold_results else 0.0

    # Metadata
    meta = {
        "timestamp": ts,
        "feature_columns": feature_cols,
        "threshold": threshold,
        "n_features": len(feature_cols),
        "n_folds": len(fold_results),
        "avg_val_loss": avg_val_loss,
        "fold_results_sampled": sampled_folds,
    }
    meta_path = save_dir / f"metadata_{ts}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Predictions CSV (rolling OOS predictions for backtest)
    if predictions_data is not None:
        pred_df = pd.DataFrame({
            "date": predictions_data["pred_dates"],
            "prediction": predictions_data["predictions"],
            "actual": predictions_data["actuals"],
        })
        pred_path = save_dir / f"predictions_{ts}.csv"
        pred_df.to_csv(pred_path, index=False)

    logger.info("Artifacts saved to %s with timestamp %s", save_dir, ts)
    return save_dir


def load_latest_artifacts(directory: Optional[Path] = None):
    """Load the most recent model, scaler, metadata, and predictions."""
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

    # Load rolling predictions (if available)
    pred_path = load_dir / f"predictions_{ts}.csv"
    predictions_df = None
    if pred_path.exists():
        predictions_df = pd.read_csv(pred_path, parse_dates=["date"])

    return model, scaler, meta, predictions_df


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
