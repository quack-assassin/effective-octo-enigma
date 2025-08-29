"""
Bollinger Bands Trading Strategy (Breakout or Mean-Reversion)
-----------------------------------------------------------------
A clean, production-friendly module to generate Bollinger Band signals
and run a simple backtest with costs. Pure pandas/numpy, no 3rd-party
TA libraries required.

Usage (TL;DR):
    import pandas as pd
    from bollinger_strategy import BBandsConfig, run_strategy, backtest

    # df must have a DatetimeIndex and a 'close' column (open/high/low optional)
    df = pd.read_csv("your_prices.csv", parse_dates=["date"], index_col="date")

    cfg = BBandsConfig(window=20, sigma=2.0, mode="mean_reversion", allow_shorts=True,
                       fee_bps=5, slippage_bps=5, stop_loss=0.03, take_profit=0.06)

    signals = run_strategy(df, cfg)  # returns df with bands, signals, positions
    equity, stats = backtest(df, signals, cfg)
    print(stats)

Notes:
- This module assumes signals are generated using bar-close information and
  positions are updated on the next bar (to reduce look-ahead bias).
- Costs are applied on position changes using a simple bps model.
- This is educational code. Not financial advice. Trade responsibly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Configuration
# ----------------------------
@dataclass
class BBandsConfig:
    window: int = 20
    sigma: float = 2.0
    mode: str = "mean_reversion"  # or "breakout"
    allow_shorts: bool = True

    # Risk controls (percent as decimals). If None, disabled.
    stop_loss: Optional[float] = None   # e.g., 0.03 -> 3%
    take_profit: Optional[float] = None # e.g., 0.06 -> 6%

    # Trading costs in basis points (1/100 of a percent).
    fee_bps: float = 0.0
    slippage_bps: float = 0.0

    # Annualization factor for daily data
    periods_per_year: int = 252

    # Column to trade on
    price_col: str = "close"

    def validate(self) -> None:
        assert self.window >= 2, "window must be >= 2"
        assert self.sigma > 0, "sigma must be > 0"
        assert self.mode in {"mean_reversion", "breakout"}, "mode must be 'mean_reversion' or 'breakout'"
        assert self.price_col, "price_col must be set"


# ----------------------------
# Indicators
# ----------------------------

def bollinger_bands(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 20,
    sigma: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands, %B and Bandwidth.

    Returns a new DataFrame with columns: mid, upper, lower, pband, bandwidth.
    """
    px = df[price_col].astype(float)
    mid = px.rolling(window=window, min_periods=window).mean()
    std = px.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + sigma * std
    lower = mid - sigma * std

    pband = (px - lower) / (upper - lower)  # aka %B
    bandwidth = (upper - lower) / mid

    out = pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "pband": pband,
            "bandwidth": bandwidth,
        },
        index=df.index,
    )
    return out


# ----------------------------
# Signal Generation (stateful)
# ----------------------------

def _generate_signals_stateful(
    df: pd.DataFrame,
    cfg: BBandsConfig,
) -> pd.DataFrame:
    """Create entry/exit signals using a small state machine.

    Positions: -1 (short), 0 (flat), +1 (long)

    Mean-Reversion mode:
      - Long entry: close crosses UP through lower band.
      - Long exit:  close crosses UP through mid (SMA) OR risk controls.
      - Short entry: close crosses DOWN through upper band.
      - Short exit:  close crosses DOWN through mid (SMA) OR risk controls.

    Breakout mode:
      - Long entry: close crosses UP through upper band (breakout).
      - Long exit:  close crosses DOWN through mid OR risk controls.
      - Short entry: close crosses DOWN through lower band (breakdown).
      - Short exit:  close crosses UP through mid OR risk controls.

    Risk controls (optional): fixed % stop and take-profit from entry.
    """
    cfg.validate()

    price = df[cfg.price_col].astype(float).values
    mid = df["mid"].values
    upper = df["upper"].values
    lower = df["lower"].values

    n = len(df)
    position = np.zeros(n, dtype=float)
    signal = np.zeros(n, dtype=float)  # +1 buy, -1 sell, 0 hold

    # Trade state
    curr_pos = 0  # -1, 0, +1
    entry_price = np.nan

    for i in range(1, n):  # start at 1 so we can use i-1 for cross detections
        px_prev, px_now = price[i - 1], price[i]
        mid_prev, mid_now = mid[i - 1], mid[i]
        up_prev, up_now = upper[i - 1], upper[i]
        lo_prev, lo_now = lower[i - 1], lower[i]

        # Skip until bands are fully formed
        if np.isnan(mid_now) or np.isnan(up_now) or np.isnan(lo_now):
            position[i] = curr_pos
            continue

        # Helper: crosses
        crosses_up = lambda prev_px, now_px, prev_lvl, now_lvl: prev_px < prev_lvl and now_px >= now_lvl
        crosses_down = lambda prev_px, now_px, prev_lvl, now_lvl: prev_px > prev_lvl and now_px <= now_lvl

        # Risk management exits
        def hit_stop_take_profit(curr_pos: int, entry: float, now_px: float) -> bool:
            if np.isnan(entry):
                return False
            # Stop loss
            if cfg.stop_loss is not None and cfg.stop_loss > 0:
                if curr_pos > 0 and now_px <= entry * (1 - cfg.stop_loss):
                    return True
                if curr_pos < 0 and now_px >= entry * (1 + cfg.stop_loss):
                    return True
            # Take profit
            if cfg.take_profit is not None and cfg.take_profit > 0:
                if curr_pos > 0 and now_px >= entry * (1 + cfg.take_profit):
                    return True
                if curr_pos < 0 and now_px <= entry * (1 - cfg.take_profit):
                    return True
            return False

        exited = False
        # Exit signals based on midline or risk
        if curr_pos > 0:
            if crosses_up(px_prev, px_now, mid_prev, mid_now) or hit_stop_take_profit(curr_pos, entry_price, px_now):
                curr_pos = 0
                signal[i] = -1  # sell to flat
                entry_price = np.nan
                exited = True
        elif curr_pos < 0:
            if crosses_down(px_prev, px_now, mid_prev, mid_now) or hit_stop_take_profit(curr_pos, entry_price, px_now):
                curr_pos = 0
                signal[i] = +1  # buy to flat
                entry_price = np.nan
                exited = True

        if not exited:
            # Entry rules
            if cfg.mode == "mean_reversion":
                # Long entry: cross back above lower band after being below
                if curr_pos == 0 and crosses_up(px_prev, px_now, lo_prev, lo_now):
                    curr_pos = +1
                    signal[i] = +1
                    entry_price = px_now
                # Short entry: cross back below upper band after being above
                elif cfg.allow_shorts and curr_pos == 0 and crosses_down(px_prev, px_now, up_prev, up_now):
                    curr_pos = -1
                    signal[i] = -1
                    entry_price = px_now

            elif cfg.mode == "breakout":
                # Long entry: cross ABOVE upper band (breakout)
                if curr_pos == 0 and crosses_up(px_prev, px_now, up_prev, up_now):
                    curr_pos = +1
                    signal[i] = +1
                    entry_price = px_now
                # Short entry: cross BELOW lower band (breakdown)
                elif cfg.allow_shorts and curr_pos == 0 and crosses_down(px_prev, px_now, lo_prev, lo_now):
                    curr_pos = -1
                    signal[i] = -1
                    entry_price = px_now

        position[i] = curr_pos

    out = df.copy()
    out["signal"] = signal  # +1 buy, -1 sell, 0 hold
    out["position"] = position  # -1,0,+1 current exposure
    return out


# ----------------------------
# Public API
# ----------------------------

def run_strategy(df: pd.DataFrame, cfg: BBandsConfig) -> pd.DataFrame:
    """Compute Bollinger Bands and generate signals/positions.

    Returns a DataFrame with columns:
      mid, upper, lower, pband, bandwidth, signal, position
    """
    df = df.copy()
    assert cfg.price_col in df.columns, f"'{cfg.price_col}' column missing"

    bands = bollinger_bands(df, price_col=cfg.price_col, window=cfg.window, sigma=cfg.sigma)
    merged = df.join(bands)
    signals = _generate_signals_stateful(merged, cfg)
    return signals


def backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: BBandsConfig,
) -> Tuple[pd.Series, Dict[str, float]]:
    """Simple vectorized backtest with trading costs.

    Assumptions:
      - Signal generated on bar i applies from bar i+1 (next bar) onward.
      - Returns are close-to-close percentage changes.
      - Costs are applied when position changes (turnover based).

    Returns
    -------
    equity_curve : pd.Series
        Cumulative return (1.0 = start) over time
    stats : Dict[str, float]
        Performance metrics
    """
    cfg.validate()

    price = df[cfg.price_col].astype(float)
    returns = price.pct_change().fillna(0.0)

    pos = signals["position"].shift(1).fillna(0.0)  # apply next bar

    # Turnover: absolute change in position
    turnover = pos.diff().abs().fillna(abs(pos.iloc[0]))
    cost_per_change = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    costs = turnover * cost_per_change

    strat_ret = pos * returns - costs
    equity = (1.0 + strat_ret).cumprod()

    # Metrics
    total_return = equity.iloc[-1] - 1.0
    n_periods = len(equity)

    ann_ret = (equity.iloc[-1]) ** (cfg.periods_per_year / max(n_periods, 1)) - 1.0 if n_periods > 1 else 0.0
    ann_vol = strat_ret.std(ddof=0) * np.sqrt(cfg.periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = drawdown.min()

    # Win rate approximated via sign of per-trade returns
    trades = turnover[turnover > 0]
    trade_count = int(trades.sum())  # rough count: each change = 1 trade action

    # Crude estimation of trade PnL: group returns by position regimes
    pnl_list = []
    last_pos = 0.0
    trade_start_idx = None
    for i in range(len(pos)):
        if i == 0:
            last_pos = pos.iloc[i]
            trade_start_idx = i if last_pos != 0 else None
            continue
        if pos.iloc[i] != last_pos:
            # trade ended at i-1
            if last_pos != 0 and trade_start_idx is not None:
                pnl = (1.0 + (returns.iloc[trade_start_idx + 1 : i] * last_pos).sum()) - 1.0
                pnl_list.append(pnl)
            # update state
            last_pos = pos.iloc[i]
            trade_start_idx = i if last_pos != 0 else None
    # close last open trade
    if last_pos != 0 and trade_start_idx is not None:
        pnl = (1.0 + (returns.iloc[trade_start_idx + 1 :] * last_pos).sum()) - 1.0
        pnl_list.append(pnl)

    pnl_arr = np.array(pnl_list)
    win_rate = float((pnl_arr > 0).mean()) if pnl_arr.size else np.nan

    stats = {
        "Total Return": float(total_return),
        "CAGR": float(ann_ret),
        "Sharpe": float(sharpe),
        "Max Drawdown": float(max_dd),
        "Win Rate": float(win_rate),
        "Trades (approx)": float(trade_count),
        "Costs (bps)": float(cfg.fee_bps + cfg.slippage_bps),
    }

    return equity, stats


# ----------------------------
# Optional: Quick self-test
# ----------------------------
if __name__ == "__main__":
    # Generate a synthetic price series for a smoke test
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    noise = np.random.normal(0, 0.005, size=n)
    drift = 0.0003
    px = 100 * (1 + drift + noise).cumprod()
    df = pd.DataFrame({"close": px}, index=dates)

    cfg = BBandsConfig(window=20, sigma=2.0, mode="mean_reversion", allow_shorts=True,
                       fee_bps=2, slippage_bps=2, stop_loss=0.05, take_profit=0.1)

    signals = run_strategy(df, cfg)
    equity, stats = backtest(df, signals, cfg)
    print("Smoke test stats:", stats)
