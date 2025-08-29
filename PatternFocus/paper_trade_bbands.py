"""
Paper Trading Runner for Bollinger Bands Strategy (Coinbase / CCXT)
-------------------------------------------------------------------
MODES:
1) Historical event-replay (default) — scans fetched candles and prints all NEW/CLOSE events at once.
2) LIVE loop (add --live true) — polls fresh candles and executes at the *open* of each new candle after a signal.

- Starting USD balance (default: 1000)
- Max concurrent open positions (default: 5)
- Risk per new position as % of current account value (default: 5%)
- Optional shorts (simulation only)
- Prints NEW/CLOSE logs exactly as requested

Run (examples):
    pip install ccxt pandas numpy

    # Historical (event replay):
    python paper_trade_bbands.py \
        --pairs XRP-USD AVAX-USD SOL-USD ETH-USD BTC-USD LINK-USD ADA-USD DOT-USD SUI-USD DOGE-USD SHIB-USD \
        --timeframe 1h --limit 1000 \
        --start-usd 1000 --max-positions 5 --risk-pct 5 \
        --mode mean_reversion --window 20 --sigma 2 --allow-shorts true

    # LIVE mode (keeps running in terminal):
    python paper_trade_bbands.py \
        --pairs XRP-USD AVAX-USD SOL-USD ETH-USD BTC-USD LINK-USD ADA-USD DOT-USD SUI-USD DOGE-USD SHIB-USD \
        --timeframe 1h --limit 200 \
        --start-usd 1000 --max-positions 5 --risk-pct 5 \
        --mode mean_reversion --window 20 --sigma 2 --allow-shorts true --live true

Notes:
- Shorts here are SIMULATED and may not reflect Coinbase spot constraints.
- We execute at the OPEN of the bar *after* a signal appears to avoid look-ahead.
- Total account value = remaining USD + sum(mark-to-market of open positions).
  For LONG: value = tokens * price.
  For SHORT: value = allocated_usd + PnL (where PnL = (entry - price) * tokens).
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ccxt  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "ccxt is required. Install with: pip install ccxt" +
        "Original error: " + str(e)
    )

from bollinger_strategy import BBandsConfig, run_strategy


# ----------------------------
# Data Fetching
# ----------------------------

def dash_to_ccxt(sym: str) -> str:
    """Convert 'ETH-USD' -> 'ETH/USD'."""
    return sym.replace("-", "/")


def fetch_ohlcv_df(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV for a symbol/timeframe and return DataFrame with datetime index.

    Columns: open, high, low, close, volume
    Index: pd.DatetimeIndex (UTC)
    """
    ex.load_markets()
    if symbol not in ex.markets:
        raise ValueError(f"Symbol {symbol} not found on exchange {ex.id}")
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        raise ValueError(f"No OHLCV returned for {symbol} {timeframe}")
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


# ----------------------------
# Portfolio & Positions
# ----------------------------

@dataclass
class Position:
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    tokens: float
    allocated_usd: float
    entry_time: pd.Timestamp

    def pnl(self, price_now: float) -> float:
        if self.side == "LONG":
            return (price_now - self.entry_price) * self.tokens
        else:  # SHORT
            return (self.entry_price - price_now) * self.tokens

    def value_on_close(self, price_now: float) -> float:
        """USD credited back to cash when closing."""
        if self.side == "LONG":
            return self.tokens * price_now
        else:
            # Return reserved margin plus PnL
            return self.allocated_usd + self.pnl(price_now)


class PaperTrader:
    def __init__(
        self,
        start_usd: float = 1000.0,
        max_positions: int = 5,
        risk_fraction: float = 0.05,
    ) -> None:
        self.cash_usd: float = float(start_usd)
        self.max_positions = int(max_positions)
        self.risk_fraction = float(risk_fraction)
        self.positions: Dict[str, Position] = {}

        # Running stats
        self.running_pl: float = 0.0
        self.closed_trades: int = 0
        self.wins: int = 0

    # ---- Helpers ----
    def _fmt(self, x: float) -> str:
        return f"{x:,.2f}"

    def _fmt_tokens(self, x: float) -> str:
        return f"{x:.8f}"

    def account_value(self, price_lookup) -> float:
        """Total USD equity given a callable price lookup(symbol)->price."""
        total = self.cash_usd
        for pos in self.positions.values():
            p = float(price_lookup(pos.symbol))
            if pos.side == "LONG":
                total += pos.tokens * p
            else:
                total += pos.allocated_usd + pos.pnl(p)
        return float(total)

    def can_open(self) -> bool:
        return len(self.positions) < self.max_positions and self.cash_usd > 0.0

    # ---- Trading Ops ----
    def open_position(self, symbol: str, side: str, entry_price: float, when: pd.Timestamp, price_lookup) -> None:
        if symbol in self.positions:
            return  # already open
        if not self.can_open():
            return

        equity_now = self.account_value(price_lookup)
        alloc = equity_now * self.risk_fraction
        alloc = min(alloc, self.cash_usd)
        if alloc <= 0:
            return
        tokens = alloc / entry_price

        # Deduct cash (lock margin for shorts as well)
        self.cash_usd -= alloc
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            entry_price=float(entry_price),
            tokens=float(tokens),
            allocated_usd=float(alloc),
            entry_time=when,
        )

        def _price(sym: str) -> float:
            return float(price_lookup(sym))

        total_val = self.account_value(_price)
        print(
            f"NEW {side} {symbol}"
            f"--- Entry Price : {self._fmt(entry_price)} --- Amount [Token] : {self._fmt_tokens(tokens)}"
            f"--- USD $ Amount : {self._fmt(alloc)}"
            f"--- Remaining USD $ To Trade : {self._fmt(self.cash_usd)}"
            f"--- Current Total Account Value in $ : {self._fmt(total_val)}"
        )

    def close_position(self, symbol: str, close_price: float, when: pd.Timestamp, price_lookup) -> None:
        pos = self.positions.get(symbol)
        if pos is None:
            return
        # Compute PnL and value returned to cash
        close_value = pos.value_on_close(close_price)
        pnl = pos.pnl(close_price)

        # Update accounting
        self.cash_usd += close_value
        self.running_pl += pnl
        self.closed_trades += 1
        if pnl > 0:
            self.wins += 1

        # Print before deletion
        win_rate = (self.wins / self.closed_trades * 100.0) if self.closed_trades else 0.0
        loss_rate = 100.0 - win_rate if self.closed_trades else 0.0

        print(
            f"CLOSE {pos.side} {symbol}"
            f"--- Entry Price : {self._fmt(pos.entry_price)} --- Amount [Token] : {self._fmt_tokens(pos.tokens)}"
            f"--- USD $ Amount on Position Open : {self._fmt(pos.allocated_usd)}"
            f"--- USD $ Amount on Position Close : {self._fmt(close_value)}"
            f"--- Profit/Loss from this Position $ : {self._fmt(pnl)}"
            f"--- Current Total P/L $ : {self._fmt(self.running_pl)}"
            f"--- Current % of Positions End in Profit : {win_rate:.2f}%"
            f"--- Current % of Positions End in Loss : {loss_rate:.2f}%"
            f"--- Remaining USD $ To Trade : {self._fmt(self.cash_usd)}"
            f"--- Current Total Account Value in $ : {self._fmt(self.account_value(price_lookup))}"
        )

        # Remove position
        del self.positions[symbol]


# ----------------------------
# Signal -> Event conversion (Historical mode)
# ----------------------------

def build_trade_events(
    signals_by_symbol: Dict[str, pd.DataFrame],
    data_by_symbol: Dict[str, pd.DataFrame],
    allow_shorts: bool,
) -> List[Tuple[pd.Timestamp, str, str, str]]:
    """
    Build chronologically ordered trade events based on position transitions.

    Returns list of tuples: (exec_time, symbol, action, side)
      - action: 'OPEN' or 'CLOSE'
      - side: 'LONG' or 'SHORT'
    Execution occurs at the *next bar's OPEN* following a transition.
    """
    events: List[Tuple[pd.Timestamp, str, str, str]] = []
    for symbol, sig in signals_by_symbol.items():
        pos = sig["position"].fillna(0.0)
        idx = sig.index
        for i in range(1, len(pos)):
            prev, curr = pos.iloc[i - 1], pos.iloc[i]
            # Determine execution index (i + 1)
            if i + 1 >= len(pos):
                continue  # no next bar to execute
            exec_time = idx[i + 1]
            # Open
            if prev == 0 and curr == 1:
                events.append((exec_time, symbol, "OPEN", "LONG"))
            elif allow_shorts and prev == 0 and curr == -1:
                events.append((exec_time, symbol, "OPEN", "SHORT"))
            # Close
            elif prev == 1 and curr == 0:
                events.append((exec_time, symbol, "CLOSE", "LONG"))
            elif allow_shorts and prev == -1 and curr == 0:
                events.append((exec_time, symbol, "CLOSE", "SHORT"))
    # Sort by time (stable for equal times)
    events.sort(key=lambda x: (x[0], x[1]))
    return events


# ----------------------------
# Price lookup helpers
# ----------------------------

def make_price_lookup(data_by_symbol: Dict[str, pd.DataFrame]):
    # Pre-cache series for speed
    open_series = {s: df["open"].sort_index() for s, df in data_by_symbol.items()}

    def price_at(sym: str, when: pd.Timestamp) -> float:
        s = open_series.get(sym)
        if s is None or s.empty:
            return float("nan")
        # Use the price at 'when' if present, else last available prior to 'when'
        s2 = s.loc[:when]
        if s2.empty:
            return float(s.iloc[0])
        return float(s2.iloc[-1])

    return price_at


# ----------------------------
# Timeframe helpers
# ----------------------------

def timeframe_to_seconds(tf: str) -> int:
    tf = tf.strip().lower()
    units = {
        'm': 60,
        'h': 3600,
        'd': 86400,
    }
    # e.g., '1h', '15m', '4h', '1d'
    num = ''.join([c for c in tf if c.isdigit()])
    unit = ''.join([c for c in tf if c.isalpha()])
    if not num or unit not in units:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return int(num) * units[unit]


# ----------------------------
# LIVE mode
# ----------------------------

def live_loop(args):
    ex = ccxt.coinbase({"enableRateLimit": True})

    # Track last executed bar per symbol to avoid duplicate actions within same bar
    last_exec_ts: Dict[str, Optional[pd.Timestamp]] = {dash_to_ccxt(p): None for p in args.pairs}

    trader = PaperTrader(
        start_usd=args.start_usd,
        max_positions=args.max_positions,
        risk_fraction=args.risk_pct / 100.0,
    )

    tf_secs = timeframe_to_seconds(args.timeframe)
    min_limit = max(args.window + 5, 60)  # ensure enough history for bands

    print(f"LIVE mode started. Timeframe={args.timeframe} (≈{tf_secs}s), polling… Press Ctrl+C to stop.")

    try:
        while True:
            data_by_symbol: Dict[str, pd.DataFrame] = {}
            for pair in args.pairs:
                sym = dash_to_ccxt(pair)
                try:
                    df = fetch_ohlcv_df(ex, sym, timeframe=args.timeframe, limit=max(args.limit, min_limit))
                    data_by_symbol[sym] = df
                except Exception as e:
                    print(f"Fetch error {pair}: {e}")
                    continue

            now_prices = {s: float(df.iloc[-1]["open"]) for s, df in data_by_symbol.items() if not df.empty}
            price_lookup = lambda s: now_prices.get(s, float("nan"))

            for sym, df in data_by_symbol.items():
                if len(df) < args.window + 3:
                    continue
                cfg = BBandsConfig(
                    window=args.window,
                    sigma=args.sigma,
                    mode=args.mode,
                    allow_shorts=args.allow_shorts,
                    price_col="close",
                )
                sig = run_strategy(df[["open","high","low","close","volume"]].copy(), cfg)
                pos = sig["position"].fillna(0.0)
                idx = sig.index
                # Use last CLOSED bar (idx[-2]) for signal, execute at current bar open (idx[-1])
                if len(idx) < 3:
                    continue
                prev_pos = pos.iloc[-3]
                curr_pos = pos.iloc[-2]
                exec_time = idx[-1]
                if last_exec_ts.get(sym) is not None and exec_time <= last_exec_ts[sym]:
                    continue  # already acted this bar

                exec_price = float(df.iloc[-1]["open"])  # open of current bar

                # CLOSE signals first
                if prev_pos == 1 and curr_pos == 0 and sym in trader.positions and trader.positions[sym].side == "LONG":
                    trader.close_position(sym, exec_price, exec_time, price_lookup)
                    last_exec_ts[sym] = exec_time
                elif args.allow_shorts and prev_pos == -1 and curr_pos == 0 and sym in trader.positions and trader.positions[sym].side == "SHORT":
                    trader.close_position(sym, exec_price, exec_time, price_lookup)
                    last_exec_ts[sym] = exec_time
                # OPEN signals
                elif prev_pos == 0 and curr_pos == 1 and sym not in trader.positions:
                    trader.open_position(sym, "LONG", exec_price, exec_time, price_lookup)
                    last_exec_ts[sym] = exec_time
                elif args.allow_shorts and prev_pos == 0 and curr_pos == -1 and sym not in trader.positions:
                    trader.open_position(sym, "SHORT", exec_price, exec_time, price_lookup)
                    last_exec_ts[sym] = exec_time

            # Sleep until next bar boundary (plus small buffer)
            now = math.floor(time.time())
            sleep_for = tf_secs - (now % tf_secs) + 1
            # Failsafe: don't sleep absurdly long if user picked huge tf
            sleep_for = max(5, min(sleep_for, tf_secs))
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        pass
    finally:
        # Final summary
        final_equity = trader.account_value(lambda s: now_prices.get(s, float("nan"))) if 'now_prices' in locals() else trader.cash_usd
        print("==== FINAL SUMMARY ====")
        print(f"Remaining USD: {trader.cash_usd:,.2f}")
        print(f"Open Positions: {len(trader.positions)}")
        for sym, pos in trader.positions.items():
            last_price = now_prices.get(sym, pos.entry_price)
            mv = pos.tokens * last_price if pos.side == "LONG" else pos.allocated_usd + pos.pnl(last_price)
            print(f" - {sym} {pos.side}: tokens={pos.tokens:.8f}, entry={pos.entry_price:.4f}, MV=${mv:,.2f}")
        print(f"Closed Trades: {trader.closed_trades}  Wins: {trader.wins}  Win%: {(trader.wins / trader.closed_trades * 100.0) if trader.closed_trades else 0.0:.2f}%")
        print(f"Running P/L: {trader.running_pl:,.2f}")
        print(f"Total Account Value: {final_equity:,.2f}")


# ----------------------------
# Historical (event replay) mode
# ----------------------------

def historical_mode(args):
    ex = ccxt.coinbase()

    # Load data & signals
    data_by_symbol: Dict[str, pd.DataFrame] = {}
    signals_by_symbol: Dict[str, pd.DataFrame] = {}

    for pair in args.pairs:
        ccxt_symbol = dash_to_ccxt(pair)
        try:
            df = fetch_ohlcv_df(ex, ccxt_symbol, timeframe=args.timeframe, limit=args.limit)
        except Exception as e:
            print(f"Skipping {pair}: {e}")
            continue
        cfg = BBandsConfig(
            window=args.window,
            sigma=args.sigma,
            mode=args.mode,
            allow_shorts=args.allow_shorts,
            price_col="close",
        )
        sig = run_strategy(df[["open","high","low","close","volume"]].copy(), cfg)
        data_by_symbol[ccxt_symbol] = df
        signals_by_symbol[ccxt_symbol] = sig

    if not data_by_symbol:
        raise SystemExit("No data loaded. Check pairs/timeframe.")

    price_lookup = make_price_lookup(data_by_symbol)

    # Build events timeline
    events = build_trade_events(signals_by_symbol, data_by_symbol, allow_shorts=args.allow_shorts)
    if not events:
        print("No trade events generated with current parameters.")
        return

    trader = PaperTrader(
        start_usd=args.start_usd,
        max_positions=args.max_positions,
        risk_fraction=args.risk_pct / 100.0,
    )

    # Process events chronologically
    for when, symbol, action, side in events:
        # Determine execution price at next-bar OPEN
        df = data_by_symbol[symbol]
        if when not in df.index:
            # If exact timestamp missing (due to sparse data), use nearest prior open
            exec_price = price_lookup(symbol, when)
        else:
            exec_price = float(df.loc[when, "open"])  # execute at open

        if np.isnan(exec_price):
            continue

        if action == "OPEN":
            trader.open_position(symbol, side, exec_price, when, lambda s: price_lookup(s, when))
        elif action == "CLOSE":
            pos = trader.positions.get(symbol)
            if pos is not None:
                trader.close_position(symbol, exec_price, when, lambda s: price_lookup(s, when))

    # Final summary
    final_equity = trader.account_value(lambda s: price_lookup(s, max([t for t, *_ in events])))
    print("==== FINAL SUMMARY ====")
    print(f"Remaining USD: {trader.cash_usd:,.2f}")
    print(f"Open Positions: {len(trader.positions)}")
    for sym, pos in trader.positions.items():
        last_price = price_lookup(sym, events[-1][0])
        mv = pos.tokens * last_price if pos.side == "LONG" else pos.allocated_usd + pos.pnl(last_price)
        print(f" - {sym} {pos.side}: tokens={pos.tokens:.8f}, entry={pos.entry_price:.4f}, MV=${mv:,.2f}")
        print(f"Closed Trades: {trader.closed_trades}  Wins: {trader.wins}  Win%: {(trader.wins / trader.closed_trades * 100.0) if trader.closed_trades else 0.0:.2f}%")
    print(f"Running P/L: {trader.running_pl:,.2f}")
    print(f"Total Account Value: {final_equity:,.2f}")


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Paper trade Bollinger Bands on Coinbase")
    p.add_argument("--pairs", nargs="+", required=True, help="Pairs like ETH-USD ...")
    p.add_argument("--timeframe", default="1h", help="CCXT timeframe (e.g., 1m, 5m, 1h, 4h, 1d)")
    p.add_argument("--limit", type=int, default=1000, help="Bars to fetch per pair")
    p.add_argument("--start-usd", type=float, default=1000.0)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--risk-pct", type=float, default=5.0, help="Percent of equity per new trade")

    # Strategy
    p.add_argument("--mode", choices=["mean_reversion", "breakout"], default="mean_reversion")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--allow-shorts", type=lambda s: s.lower() in {"1","true","yes","y"}, default=False)

    # LIVE toggle
    p.add_argument("--live", type=lambda s: s.lower() in {"1","true","yes","y"}, default=False, help="Run in live polling mode")

    args = p.parse_args()

    if args.live:
        live_loop(args)
    else:
        historical_mode(args)


if __name__ == "__main__":
    main()
