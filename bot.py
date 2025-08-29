#!/usr/bin/env python3
"""
US Day-Trading Bot — EMA + MACD + ATR with Volume Filter
=========================================================

Legal/Compliance (USA):
- Connects to a **broker/exchange API** via ccxt using API keys. No custody of funds.
- Defaults to **paper trading**; switch to live only after testing.
- Designed for **U.S.-friendly exchanges** (Coinbase Exchange, Kraken, Gemini). Avoid offshore/leverage.
- Always restrict API keys to **trade only** (no withdrawals).

Dependencies:
    pip install ccxt pandas numpy python-dotenv

Environment Variables (set only what your exchange needs):
    EXCHANGE_ID=coinbase|kraken|gemini
    EXCHANGE_API_KEY=...
    EXCHANGE_SECRET=...
    EXCHANGE_PASSWORD=...         # (Kraken: optional, Coinbase: passphrase)
    EXCHANGE_PASSPHRASE=...       # (Coinbase specific)

Run:
    # Backtest last 30 days on BTC/USD 1m candles
    python "US Day-Trading Bot (EMA+MACD+ATR).py" --mode backtest --symbol BTC/USD --timeframe 1m --days 30

    # Paper-trade live (single symbol)
    python "US Day-Trading Bot (EMA+MACD+ATR).py" --mode paper --exchange coinbase --symbol BTC/USD --timeframe 1m

    # Paper-trade a **universe** (top 30 USD pairs by 24h volume on exchange)
    python "US Day-Trading Bot (EMA+MACD+ATR).py" --mode paper --exchange coinbase --universe top30usd --timeframe 1m --equity 100 --risk 5

    # Live trading (AFTER TESTING!)
    python "US Day-Trading Bot (EMA+MACD+ATR).py" --mode live --exchange coinbase --symbol BTC/USD --timeframe 1m

Notes:
- Strategy is **long-only** day trading. You can extend for shorts if your broker and rules allow.
- Uses **EMA trend filter (50>200)** + **MACD bullish cross** + **volume spike** + **ATR risk**.
- Risk mgmt: fixed % risk per trade, ATR stop, optional trailing stop, daily loss cap.
"""

from __future__ import annotations
import argparse
import asyncio
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

# ------------------------------
# Helpers: Indicators
# ------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ------------------------------
# Config
# ------------------------------

US_FRIENDLY_EXCHANGES = {
    'coinbase',      # Coinbase Exchange (formerly Pro)
    'kraken',
    'gemini',
}

@dataclass
class StrategyConfig:
    timeframe: str = '1m'
    symbol: str = 'BTC/USD'

    # Core indicators
    ema_fast: int = 50
    ema_slow: int = 200
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vol_ma_period: int = 20
    vol_min_mult: float = 1.2   # require volume > 1.2x avg
    atr_period: int = 14

    # Risk & trade mgmt
    equity: float = 10000.0      # used for paper/backtest sizing
    risk_per_trade_pct: float = 0.01  # risk 1% of equity per trade
    atr_stop_mult: float = 1.5
    take_profit_rr: float = 2.0      # take profit at 2R (twice the risk)
    trailing_atr_mult: Optional[float] = 1.0  # trail by 1x ATR; set None to disable
    max_daily_loss_pct: float = 0.03  # stop trading for day at -3%
    max_positions: int = 1

    # Backtest / live
    days_backtest: int = 30

# ------------------------------
# Exchange Wrapper
# ------------------------------

class ExchangeClient:
    def __init__(self, exchange_id: str, mode: str):
        if exchange_id not in US_FRIENDLY_EXCHANGES:
            raise ValueError(f"Exchange '{exchange_id}' is not in US-friendly allowlist {US_FRIENDLY_EXCHANGES}")
        self.exchange_id = exchange_id
        self.mode = mode  # backtest|paper|live

        load_dotenv()
        api_key = os.getenv('EXCHANGE_API_KEY')
        secret = os.getenv('EXCHANGE_SECRET')
        password = os.getenv('EXCHANGE_PASSWORD') or os.getenv('EXCHANGE_PASSPHRASE')

        klass = getattr(ccxt, exchange_id)
        params = {
            'apiKey': api_key,
            'secret': secret,
        }
        if exchange_id == 'coinbase' and password:
            params['password'] = password
        if exchange_id == 'kraken' and password:
            params['password'] = password

        self.client = klass(params)
        self.client.load_markets()

    def ohlcv(self, symbol: str, timeframe: str, limit: int = 500, since: Optional[int] = None) -> List[List[float]]:
        return self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)

    def price(self, symbol: str) -> float:
        ticker = self.client.fetch_ticker(symbol)
        return float(ticker['last'])

    # Live order endpoints are included but guarded by mode
    def create_market_buy(self, symbol: str, amount: float):
        if self.mode != 'live':
            return {'id': 'paper-buy', 'amount': amount}
        return self.client.create_order(symbol, 'market', 'buy', amount)

    def create_market_sell(self, symbol: str, amount: float):
        if self.mode != 'live':
            return {'id': 'paper-sell', 'amount': amount}
        return self.client.create_order(symbol, 'market', 'sell', amount)

# ------------------------------
# Strategy Core
# ------------------------------

class EMAMACDATRStrategy:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        # positions keyed by symbol for multi-asset support
        # position dict: {'side': 'long', 'entry': float, 'size': float, 'stop': float, 'trail': float, 'initial_stop': float}
        self.positions: Dict[str, Dict] = {}
        self.daily_pl: float = 0.0
        self.last_trade_day: Optional[datetime] = None

    def build_frame(self, raw: List[List[float]]) -> pd.DataFrame:
        df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_convert('America/Chicago')
        df.set_index('ts', inplace=True)
        # Indicators
        df['ema_fast'] = ema(df['close'], self.cfg.ema_fast)
        df['ema_slow'] = ema(df['close'], self.cfg.ema_slow)
        macd_line, signal_line, hist = macd(df['close'], self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal)
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = hist
        df['atr'] = atr(df, self.cfg.atr_period)
        df['vol_ma'] = df['volume'].rolling(self.cfg.vol_ma_period).mean()
        return df.dropna()

    def long_signal(self, df: pd.DataFrame) -> bool:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        trend_ok = last['ema_fast'] > last['ema_slow']
        macd_cross_up = (prev['macd'] <= prev['macd_signal']) and (last['macd'] > last['macd_signal'])
        volume_ok = last['volume'] > self.cfg.vol_min_mult * last['vol_ma']
        price_above_fast = last['close'] > last['ema_fast']
        return trend_ok and macd_cross_up and volume_ok and price_above_fast

    def size_position(self, equity: float, entry: float, stop: float) -> float:
        risk_per_trade = equity * self.cfg.risk_per_trade_pct
        per_unit_risk = max(entry - stop, 1e-8)
        units = risk_per_trade / per_unit_risk
        return max(0.0, units)

    def maybe_reset_daily(self, now: datetime):
        if self.last_trade_day is None or now.date() != self.last_trade_day.date():
            self.daily_pl = 0.0
            self.last_trade_day = now

    def manage_position(self, bar: pd.Series, pos: Dict) -> Tuple[Optional[str], Optional[float]]:
        if not pos:
            return None, None
        # Update trailing stop
        if self.cfg.trailing_atr_mult is not None:
            trail = bar['close'] - self.cfg.trailing_atr_mult * bar['atr']
            pos['trail'] = max(pos.get('trail', -math.inf), trail)
            pos['stop'] = max(pos['stop'], pos['trail'])
        # Take profit at R multiple
        risk = pos['entry'] - pos['initial_stop']
        tp_price = pos['entry'] + self.cfg.take_profit_rr * risk
        if bar['high'] >= tp_price:
            return 'take_profit', tp_price
        # Stop loss
        if bar['low'] <= pos['stop']:
            return 'stop_loss', pos['stop']
        return None, None

    def on_new_bar(self, symbol: str, df: pd.DataFrame, equity: float) -> Tuple[Optional[str], Dict]:
        bar = df.iloc[-1]
        now = df.index[-1]
        self.maybe_reset_daily(now)
        details: Dict = {'symbol': symbol}

        # Daily loss guard
        if self.daily_pl <= -self.cfg.max_daily_loss_pct * equity:
            return None, {'reason': 'daily_loss_limit_reached'}

        # Manage open position for this symbol
        pos = self.positions.get(symbol)
        if pos:
            exit_reason, exit_price = self.manage_position(bar, pos)
            if exit_reason:
                pnl = (exit_price - pos['entry']) * pos['size']
                self.daily_pl += pnl
                details.update({'exit_reason': exit_reason, 'exit_price': exit_price, 'pnl': pnl})
                self.positions.pop(symbol, None)
                return 'sell', details
            return None, {}

        # Respect max positions across portfolio
        if len(self.positions) >= self.cfg.max_positions:
            return None, {'reason': 'max_positions_reached'}

        # Entry logic
        if self.long_signal(df):
            entry = float(bar['close'])
            stop = entry - self.cfg.atr_stop_mult * float(bar['atr'])
            size = self.size_position(equity, entry, stop)
            if size <= 0:
                return None, {'reason': 'size_zero'}
            self.positions[symbol] = {
                'side': 'long',
                'entry': entry,
                'size': size,
                'initial_stop': stop,
                'stop': stop,
                'trail': stop,
            }
            details.update({'entry': entry, 'stop': stop, 'size': size})
            return 'buy', details

        return None, {}
# ------------------------------
# Backtester / Runner
# ------------------------------

def backtest(exchange: ExchangeClient, cfg: StrategyConfig) -> None:
    # Single-symbol backtest remains (universe backtest omitted for brevity)
    limit = 3000 if cfg.timeframe.endswith('m') else 1500
    raw = exchange.ohlcv(cfg.symbol, cfg.timeframe, limit=limit)
    strat = EMAMACDATRStrategy(cfg)
    df = strat.build_frame(raw)

    equity = cfg.equity
    equity_curve = []
    trades = []

    for i in range(max(cfg.ema_slow, cfg.atr_period, cfg.vol_ma_period) + 1, len(df)):
        window = df.iloc[: i + 1]
        action, info = strat.on_new_bar(cfg.symbol, window, equity)
        if action == 'buy':
            trades.append({'time': window.index[-1], 'symbol': cfg.symbol, 'type': 'BUY', **info})
        elif action == 'sell':
            equity += info.get('pnl', 0.0)
            trades.append({'time': window.index[-1], 'symbol': cfg.symbol, 'type': 'SELL', **info})
        equity_curve.append({'time': window.index[-1], 'equity': equity})

    curve = pd.DataFrame(equity_curve).set_index('time')
    tr = pd.DataFrame(trades)

    total_return = (equity / cfg.equity - 1) * 100
    max_dd = (curve['equity'].cummax() - curve['equity']).max()

    print("=== Backtest Summary ===")
    print(f"Symbol: {cfg.symbol}  TF: {cfg.timeframe}  Bars: {len(df)}")
    print(f"Start Equity: ${cfg.equity:,.2f}  End Equity: ${equity:,.2f}  Return: {total_return:,.2f}%")
    print(f"Max Drawdown (absolute $): ${max_dd:,.2f}")
    if not tr.empty:
        sells = tr[tr['type']=='SELL']
        wins = sells['pnl'] > 0
        win_rate = 100 * wins.sum() / max(1, len(sells))
        avg_win = sells.loc[wins, 'pnl'].mean() if wins.any() else 0.0
        avg_loss = sells.loc[~wins, 'pnl'].mean() if (~wins).any() else 0.0
        print(f"Trades: {len(tr)}  Win rate: {win_rate:,.2f}%  Avg Win: ${avg_win:,.2f}  Avg Loss: ${avg_loss:,.2f}")
    else:
        print("No trades.")
# ------------------------------
# Live / Paper Runner
# ------------------------------

async def pick_universe(exchange: ExchangeClient, mode: str, universe: Optional[str], explicit_symbols: Optional[str], timeframe: str) -> List[str]:
    """Decide which symbols to trade.
    - If explicit_symbols provided (comma-separated), use those.
    - If universe == 'top30usd', query tickers and pick top 30 USD quote pairs by 24h volume.
    - Else, fall back to cfg.symbol single.
    """
    if explicit_symbols:
        return [s.strip() for s in explicit_symbols.split(',') if s.strip()]

    if universe == 'top30usd':
        try:
            tickers = exchange.client.fetch_tickers()
            candidates = []
            for sym, t in tickers.items():
                if not sym.endswith('/USD'):
                    continue
                vol = t.get('quoteVolume') or t.get('baseVolume') or 0
                candidates.append((sym, vol))
            top = [s for s,_ in sorted(candidates, key=lambda x: (x[1] or 0), reverse=True)[:30]]
            # Ensure we have market data support
            markets = exchange.client.load_markets()
            top = [s for s in top if s in markets]
            return top if top else []
        except Exception:
            return []
    return []

async def run_loop(exchange: ExchangeClient, cfg: StrategyConfig, universe: Optional[str] = None, symbols_csv: Optional[str] = None):
    strat = EMAMACDATRStrategy(cfg)
    equity = cfg.equity  # paper mode equity

    symbols = await pick_universe(exchange, exchange.mode, universe, symbols_csv, cfg.timeframe)
    if not symbols:
        symbols = [cfg.symbol]

    frames: Dict[str, pd.DataFrame] = {}

    print(f"Starting {exchange.exchange_id.upper()} {cfg.timeframe} mode={exchange.mode} symbols={symbols}")
    while True:
        try:
            # refresh data per symbol
            for sym in symbols:
                raw = exchange.ohlcv(sym, cfg.timeframe, limit=600)
                frames[sym] = strat.build_frame(raw)

            # evaluate each symbol
            for sym, df in frames.items():
                action, info = strat.on_new_bar(sym, df, equity)
                if action == 'buy':
                    print(f"BUY {sym} @ {info['entry']:.4f} size={info['size']:.6f} stop={info['stop']:.4f}")
                elif action == 'sell':
                    equity += info.get('pnl', 0.0)
                    print(f"SELL {sym} ({info.get('exit_reason')}) @ {info.get('exit_price'):.4f} pnl=${info.get('pnl',0.0):.2f} daily_pl=${strat.daily_pl:.2f}")

            await asyncio.sleep(5)
        except Exception as e:
            print(f"Loop error: {e}")
            await asyncio.sleep(10)
# ------------------------------
# CLI
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='US Day-Trading Bot (EMA+MACD+ATR)')
    p.add_argument('--mode', choices=['backtest','paper','live'], default='paper')
    p.add_argument('--exchange', default=os.getenv('EXCHANGE_ID','coinbase'))
    p.add_argument('--symbol', default='BTC/USD', help='Fallback single symbol if no universe provided')
    p.add_argument('--symbols', default=None, help='Comma-separated list of symbols, e.g., "BTC/USD,ETH/USD,SOL/USD"')
    p.add_argument('--universe', default=None, help='Use preset universe, e.g., top30usd')
    p.add_argument('--timeframe', default='1m')
    p.add_argument('--days', type=int, default=30)
    p.add_argument('--equity', type=float, default=10000.0)
    p.add_argument('--risk', type=float, default=1.0, help='Risk per trade in % of equity')
    p.add_argument('--max-dday', type=float, default=3.0, help='Max daily loss in % to stop trading')
    p.add_argument('--max-positions', type=int, default=1, help='Max concurrent open positions across symbols')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()

    cfg = StrategyConfig(
        timeframe=args.timeframe,
        symbol=args.symbol,
        equity=args.equity,
        risk_per_trade_pct=args.risk/100.0,
        max_daily_loss_pct=args.max_dday/100.0,
        max_positions=args.max_positions,
    )

    ex = ExchangeClient(args.exchange, args.mode)

    if args.mode == 'backtest':
        backtest(ex, cfg)
    else:
        asyncio.run(run_loop(ex, cfg, universe=args.universe, symbols_csv=args.symbols))
