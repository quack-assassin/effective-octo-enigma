#!/usr/bin/env python3
"""
Pattern-Focused Day-Trading Bot — v3
- Pattern entries (hammer, bullish engulfing, doji, RSI bullish divergence, flag/pennant)
- Pattern-based stop with buffer (default 1%)
- Paper fills emulate fees/slippage and now enforce CASH + NOTIONAL CAPS
- Rate-limit friendly polling and round‑robin symbol fetching

Install:  pip install ccxt pandas numpy python-dotenv
"""
from __future__ import annotations
import argparse
import asyncio
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

# ------------------------------
# Indicators & Pattern helpers
# ------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# Candles

def is_hammer(df: pd.DataFrame, lookback: int = 1) -> bool:
    c = df.iloc[-lookback]; body = abs(c['close']-c['open']); rng = c['high']-c['low']
    if rng <= 0: return False
    lower = min(c['open'], c['close']) - c['low']; upper = c['high'] - max(c['open'], c['close'])
    return (lower > 2*body) and (upper < body)

def is_shooting_star(df: pd.DataFrame, lookback: int = 1) -> bool:
    c = df.iloc[-lookback]; body = abs(c['close']-c['open']); rng = c['high']-c['low']
    if rng <= 0: return False
    upper = c['high'] - max(c['open'], c['close']); lower = min(c['open'], c['close']) - c['low']
    return (upper > 2*body) and (lower < body)

def is_bull_engulf(df: pd.DataFrame) -> bool:
    a, b = df.iloc[-2], df.iloc[-1]
    return (a['close'] < a['open']) and (b['close'] > b['open']) and (b['close'] >= a['open']) and (b['open'] <= a['close'])

def is_bear_engulf(df: pd.DataFrame) -> bool:
    a, b = df.iloc[-2], df.iloc[-1]
    return (a['close'] > a['open']) and (b['close'] < b['open']) and (b['close'] <= a['open']) and (b['open'] >= a['close'])

def is_doji(df: pd.DataFrame, thresh: float = 0.1) -> bool:
    c = df.iloc[-1]; rng = max(1e-8, c['high']-c['low']); body = abs(c['close']-c['open'])
    return (body / rng) < thresh

# RSI & divergences

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff(); up = delta.clip(lower=0).rolling(period).mean(); down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down + 1e-9); return 100 - (100 / (1 + rs))

def bullish_divergence(df: pd.DataFrame, period: int = 14) -> bool:
    rs = rsi(df['close'], period); p = df['close']
    lows = p.rolling(3).apply(lambda s: s.iloc[1] if s.iloc[1] == s.min() else np.nan, raw=False).dropna()
    if len(lows) < 2: return False
    i1, i2 = lows.index[-2], lows.index[-1]
    return (p.loc[i2] < p.loc[i1]) and (rs.loc[i2] > rs.loc[i1])

def bearish_divergence(df: pd.DataFrame, period: int = 14) -> bool:
    rs = rsi(df['close'], period); p = df['close']
    highs = p.rolling(3).apply(lambda s: s.iloc[1] if s.iloc[1] == s.max() else np.nan, raw=False).dropna()
    if len(highs) < 2: return False
    i1, i2 = highs.index[-2], highs.index[-1]
    return (p.loc[i2] > p.loc[i1]) and (rs.loc[i2] < rs.loc[i1])

# Flags / pennants

def flag_or_pennant_breakout(df: pd.DataFrame, lookback_trend: int = 20, lookback_flag: int = 8, range_mult: float = 0.7) -> bool:
    recent = df.iloc[-(lookback_trend + lookback_flag + 1):]
    if len(recent) < (lookback_trend + lookback_flag + 1): return False
    trend = recent.iloc[:lookback_trend]; cons = recent.iloc[lookback_trend:-1]; last = recent.iloc[-1]
    up_move = (trend['close'].iloc[-1] - trend['close'].iloc[0]) / max(1e-9, trend['close'].iloc[0])
    cons_range = cons['high'].max() - cons['low'].min(); atr_mean = atr(recent, 14).iloc[-2]
    breakout = last['close'] > cons['high'].max()
    return (up_move > 0.05) and (cons_range < range_mult * atr_mean) and breakout

def flag_breakout_level(df: pd.DataFrame, lookback_trend: int = 20, lookback_flag: int = 8, range_mult: float = 0.7) -> Optional[float]:
    recent = df.iloc[-(lookback_trend + lookback_flag + 1):]
    if len(recent) < (lookback_trend + lookback_flag + 1): return None
    trend = recent.iloc[:lookback_trend]; cons = recent.iloc[lookback_trend:-1]; last = recent.iloc[-1]
    up_move = (trend['close'].iloc[-1] - trend['close'].iloc[0]) / max(1e-9, trend['close'].iloc[0])
    cons_range = cons['high'].max() - cons['low'].min(); atr_mean = atr(recent, 14).iloc[-2]
    breakout = last['close'] > cons['high'].max()
    if (up_move > 0.05) and (cons_range < range_mult * atr_mean) and breakout:
        return float(cons['low'].min())
    return None

def bullish_divergence_level(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    rs = rsi(df['close'], period); p = df['close']
    lows = p.rolling(3).apply(lambda s: s.iloc[1] if s.iloc[1] == s.min() else np.nan, raw=False).dropna()
    if len(lows) < 2: return None
    idx1, idx2 = lows.index[-2], lows.index[-1]
    if (p.loc[idx2] < p.loc[idx1]) and (rs.loc[idx2] > rs.loc[idx1]):
        return float(p.loc[idx2])
    return None

# ------------------------------
# Config
# ------------------------------

US_FRIENDLY_EXCHANGES = {'coinbase','kraken','gemini'}

@dataclass
class StrategyConfig:
    timeframe: str = '1m'
    symbol: str = 'BTC/USD'

    vol_ma_period: int = 20
    vol_min_mult: float = 1.0
    atr_period: int = 14

    equity: float = 100.0
    risk_per_trade_pct: float = 0.05  # 5% default for your setup
    take_profit_rr: float = 2.0
    trailing_atr_mult: Optional[float] = 1.0
    max_daily_loss_pct: float = 0.03
    max_positions: int = 2

    pattern_names: Tuple[str, ...] = ('hammer','bull_engulf','doji','bull_div','flag_breakout')
    pattern_confirm_volume: bool = True
    bearish_exit_on_pattern: bool = True
    pattern_stop_buffer_pct: float = 0.01  # 1% below invalidation

    # Paper fills
    fee_bps_per_side: float = 20.0
    slippage_bps: float = 5.0
    min_notional_usd: float = 5.0
    max_notional_pct_of_equity: float = 0.5  # 50% cap per order

    days_backtest: int = 30

# ------------------------------
# Exchange wrapper
# ------------------------------

class ExchangeClient:
    def __init__(self, exchange_id: str, mode: str):
        if exchange_id not in US_FRIENDLY_EXCHANGES:
            raise ValueError(f"Exchange '{exchange_id}' not in allowlist {US_FRIENDLY_EXCHANGES}")
        self.exchange_id = exchange_id; self.mode = mode
        load_dotenv(); api_key = os.getenv('EXCHANGE_API_KEY'); secret = os.getenv('EXCHANGE_SECRET')
        password = os.getenv('EXCHANGE_PASSWORD') or os.getenv('EXCHANGE_PASSPHRASE')
        klass = getattr(ccxt, exchange_id)
        params = {'apiKey': api_key, 'secret': secret, 'enableRateLimit': True, 'timeout': 30000}
        if exchange_id in {'coinbase','kraken'} and password:
            params['password'] = password
        self.client = klass(params); self.client.load_markets()

    def ohlcv(self, symbol: str, timeframe: str, limit: int = 500, since: Optional[int] = None) -> List[List[float]]:
        return self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)

# ------------------------------
# Strategy
# ------------------------------

class PatternStrategy:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.positions: Dict[str, Dict] = {}
        self.daily_pl: float = 0.0
        self.last_trade_day: Optional[datetime] = None
        # Cash accounting (paper)
        self.cash: float = cfg.equity

    def build_frame(self, raw: List[List[float]]) -> pd.DataFrame:
        df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df = df.set_index(pd.DatetimeIndex(df['ts'])).drop(columns=['ts'])
        df['atr'] = atr(df, self.cfg.atr_period)
        df['vol_ma'] = df['volume'].rolling(self.cfg.vol_ma_period).mean()
        return df.dropna()

    # returns (pattern_name, invalidation_level)
    def detect_long_pattern(self, df: pd.DataFrame) -> Optional[Tuple[str, float]]:
        last = df.iloc[-1]
        if self.cfg.pattern_confirm_volume:
            if not (last['volume'] > max(1.0, self.cfg.vol_min_mult) * last['vol_ma']):
                return None
        names = self.cfg.pattern_names
        if 'hammer' in names and is_hammer(df):
            return ('hammer', float(last['low']))
        if 'bull_engulf' in names and is_bull_engulf(df):
            return ('bull_engulf', float(last['low']))
        if 'flag_breakout' in names:
            lvl = flag_breakout_level(df)
            if lvl is not None: return ('flag_breakout', float(lvl))
        if 'bull_div' in names:
            lvl = bullish_divergence_level(df)
            if lvl is not None: return ('bull_div', float(lvl))
        if 'doji' in names and is_doji(df):
            return ('doji', float(last['low']))
        return None

    def size_position(self, equity: float, entry: float, stop: float) -> float:
        risk_per_trade = equity * self.cfg.risk_per_trade_pct
        per_unit_risk = max(entry - stop, 1e-8)
        units = risk_per_trade / per_unit_risk
        return max(0.0, units)

    def maybe_reset_daily(self, now: datetime):
        if self.last_trade_day is None or now.date() != self.last_trade_day.date():
            self.daily_pl = 0.0; self.last_trade_day = now

    def manage_position(self, bar: pd.Series, pos: Dict) -> Tuple[Optional[str], Optional[float]]:
        if not pos: return None, None
        if self.cfg.trailing_atr_mult is not None:
            trail = bar['close'] - self.cfg.trailing_atr_mult * bar['atr']
            pos['trail'] = max(pos.get('trail', -math.inf), trail)
            pos['stop'] = max(pos['stop'], pos['trail'])
        risk = pos['entry'] - pos['initial_stop']
        tp_price = pos['entry'] + self.cfg.take_profit_rr * risk
        if bar['high'] >= tp_price:
            return 'take_profit', tp_price
        if bar['low'] <= pos['stop']:
            return 'stop_loss', pos['stop']
        return None, None

    def on_new_bar(self, symbol: str, df: pd.DataFrame, equity: float) -> Tuple[Optional[str], Dict]:
        bar = df.iloc[-1]; now = df.index[-1]
        self.maybe_reset_daily(now); details: Dict = {'symbol': symbol}

        if self.daily_pl <= -self.cfg.max_daily_loss_pct * equity:
            return None, {'reason': 'daily_loss_limit_reached'}

        pos = self.positions.get(symbol)
        if pos:
            if self.cfg.bearish_exit_on_pattern:
                alerts = []
                if is_shooting_star(df): alerts.append('shooting_star')
                if is_bear_engulf(df): alerts.append('bear_engulf')
                if bearish_divergence(df): alerts.append('bearish_divergence')
                if alerts:
                    raw_exit = float(bar['close'])
                    slip = self.cfg.slippage_bps/10000.0; fee_rate = self.cfg.fee_bps_per_side/10000.0
                    exit_fill = raw_exit * (1 - slip)
                    exit_fee = exit_fill * pos['size'] * fee_rate
                    pnl = (exit_fill - pos['entry']) * pos['size'] - (pos.get('fees_paid',0.0) + exit_fee)
                    self.daily_pl += pnl
                    self.cash += exit_fill * pos['size'] - exit_fee
                    self.positions.pop(symbol, None)
                    return 'sell', {'symbol': symbol, 'exit_reason': 'bearish_pattern', 'alerts': alerts, 'exit_price': exit_fill, 'pnl': pnl, 'exit_fee': exit_fee, 'cash_remaining': self.cash}

            exit_reason, exit_price = self.manage_position(bar, pos)
            if exit_reason:
                slip = self.cfg.slippage_bps/10000.0; fee_rate = self.cfg.fee_bps_per_side/10000.0
                exit_fill = float(exit_price) * (1 - slip)
                exit_fee = exit_fill * pos['size'] * fee_rate
                pnl = (exit_fill - pos['entry']) * pos['size'] - (pos.get('fees_paid',0.0) + exit_fee)
                self.daily_pl += pnl
                self.cash += exit_fill * pos['size'] - exit_fee
                details.update({'exit_reason': exit_reason, 'exit_price': exit_fill, 'pnl': pnl, 'exit_fee': exit_fee, 'cash_remaining': self.cash})
                self.positions.pop(symbol, None)
                return 'sell', details
            return None, {}

        if len(self.positions) >= self.cfg.max_positions:
            return None, {'reason': 'max_positions_reached'}

        sig = self.detect_long_pattern(df)
        if sig:
            pat, ref = sig
            planned_entry = float(bar['close'])
            stop = min(planned_entry - 1e-6, ref * (1.0 - self.cfg.pattern_stop_buffer_pct))
            size = self.size_position(equity, planned_entry, stop)
            if size <= 0: return None, {'reason': 'size_zero'}

            # Apply paper slippage/fees
            slip = self.cfg.slippage_bps/10000.0; fee_rate = self.cfg.fee_bps_per_side/10000.0
            entry_fill = planned_entry * (1 + slip)

            # Notional caps: min-notional, % equity, and available cash
            size_cash_cap = self.cash / entry_fill if entry_fill > 0 else 0.0
            size_notional_cap = (self.cfg.max_notional_pct_of_equity * equity) / entry_fill if entry_fill > 0 else 0.0
            size_final = min(size, size_cash_cap, size_notional_cap)
            if size_final <= 0: return None, {'reason': 'insufficient_cash_or_cap'}

            notional = entry_fill * size_final
            if notional < self.cfg.min_notional_usd:
                return None, {'reason': 'min_notional', 'notional': notional}

            entry_fee = notional * fee_rate
            self.cash -= (notional + entry_fee)

            self.positions[symbol] = {
                'side': 'long', 'entry': entry_fill, 'size': size_final,
                'initial_stop': stop, 'stop': stop, 'trail': stop,
                'pattern': pat, 'ref_level': ref, 'fees_paid': entry_fee,
            }
            details.update({'entry': entry_fill, 'stop': stop, 'size': size_final, 'pattern': pat, 'ref_level': ref, 'entry_fee': entry_fee, 'notional': notional, 'cash_remaining': self.cash})
            return 'buy', details

        return None, {}

# ------------------------------
# Backtest / Live runners
# ------------------------------

def backtest(exchange: ExchangeClient, cfg: StrategyConfig) -> None:
    limit = 3000 if cfg.timeframe.endswith('m') else 1500
    raw = exchange.ohlcv(cfg.symbol, cfg.timeframe, limit=limit)
    strat = PatternStrategy(cfg)
    df = strat.build_frame(raw)

    equity = cfg.equity; equity_curve = []; trades = []
    warmup = max(cfg.atr_period, cfg.vol_ma_period) + 1
    for i in range(warmup, len(df)):
        window = df.iloc[:i+1]
        action, info = strat.on_new_bar(cfg.symbol, window, equity)
        if action == 'buy': trades.append({'time': window.index[-1], 'type':'BUY', **info})
        elif action == 'sell': equity += info.get('pnl',0.0); trades.append({'time': window.index[-1], 'type':'SELL', **info})
        equity_curve.append({'time': window.index[-1], 'equity': equity})

    curve = pd.DataFrame(equity_curve).set_index('time'); tr = pd.DataFrame(trades)
    total_return = (equity / cfg.equity - 1) * 100
    max_dd = (curve['equity'].cummax() - curve['equity']).max()
    print("=== Backtest Summary ===")
    print(f"Symbol: {cfg.symbol}  TF: {cfg.timeframe}  Bars: {len(df)}")
    print(f"Start Equity: ${cfg.equity:,.2f}  End Equity: ${equity:,.2f}  Total Return: {total_return:,.2f}%")
    print(f"Maximum Drawdown ($): ${max_dd:,.2f}")

async def pick_universe(exchange: ExchangeClient, universe: Optional[str], explicit_symbols: Optional[str], top_k: int = 30) -> List[str]:
    if explicit_symbols:
        return [s.strip() for s in explicit_symbols.split(',') if s.strip()]
    if universe == 'top30usd':
        try:
            tickers = exchange.client.fetch_tickers(); cands = []
            for sym, t in tickers.items():
                if not sym.endswith('/USD'): continue
                vol = t.get('quoteVolume') or t.get('baseVolume') or 0
                cands.append((sym, vol))
            top = [s for s,_ in sorted(cands, key=lambda x: (x[1] or 0), reverse=True)[:top_k]]
            markets = exchange.client.load_markets(); return [s for s in top if s in markets]
        except Exception: return []
    return []

async def run_loop(exchange: ExchangeClient, cfg: StrategyConfig, universe: Optional[str] = None, symbols_csv: Optional[str] = None, poll_seconds: int = 10, universe_chunk: Optional[int] = None, top_k: int = 30):
    strat = PatternStrategy(cfg); equity = cfg.equity
    symbols = await pick_universe(exchange, universe, symbols_csv, top_k)
    if not symbols: symbols = [cfg.symbol]

    frames: Dict[str, pd.DataFrame] = {}; rot = 0; total = len(symbols)
    print(f"Starting {exchange.exchange_id.upper()} {cfg.timeframe} mode={exchange.mode} symbols={symbols}")
    while True:
        try:
            active = symbols if not universe_chunk or universe_chunk >= total else symbols[rot:rot+universe_chunk]
            if universe_chunk and universe_chunk < total: rot = (rot + universe_chunk) % total
            for sym in active:
                raw = exchange.ohlcv(sym, cfg.timeframe, limit=600); frames[sym] = strat.build_frame(raw)
            for sym, df in frames.items():
                action, info = strat.on_new_bar(sym, df, equity)
                if action == 'buy':
                    print(f"BUY {sym} @ {info['entry']:.8f} | Size: {info['size']:.6f} | Stop: {info['stop']:.8f} | Pattern: {info.get('pattern')} | Reference Level: {info.get('ref_level')} | Total USD Value: ${info.get('notional', info['entry']*info['size']):,.2f} | Entry Fee: ${info.get('entry_fee',0):,.2f} | Cash Remaining: ${info.get('cash_remaining',0):,.2f}")
                elif action == 'sell':
                    equity += info.get('pnl',0.0)
                    reason_map = {'take_profit':'Take Profit','stop_loss':'Stop Loss','bearish_pattern':'Bearish Pattern Exit'}
                    rn = reason_map.get(info.get('exit_reason'), str(info.get('exit_reason')))
                    print(f"SELL {sym} ({rn}) @ {info.get('exit_price'):.8f} | Profit/Loss: ${info.get('pnl',0.0):,.2f} | Exit Fee: ${info.get('exit_fee',0.0):,.2f} | Daily Profit/Loss: ${strat.daily_pl:,.2f} | Cash Remaining: ${info.get('cash_remaining',0):,.2f}")
            await asyncio.sleep(poll_seconds)
        except Exception as e:
            print(f"Loop error: {e}"); await asyncio.sleep(30 if '429' in str(e) else 10)

# ------------------------------
# CLI
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Pattern-Focused Day-Trading Bot v3')
    p.add_argument('--mode', choices=['backtest','paper','live'], default='paper')
    p.add_argument('--exchange', default=os.getenv('EXCHANGE_ID','coinbase'))
    p.add_argument('--symbol', default='BTC/USD')
    p.add_argument('--symbols', default=None)
    p.add_argument('--universe', default=None)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--timeframe', default='1m')
    p.add_argument('--days', type=int, default=30)
    p.add_argument('--equity', type=float, default=100.0)
    p.add_argument('--risk', type=float, default=5.0, help='% of equity risked per trade')
    p.add_argument('--max-dday', type=float, default=3.0)
    p.add_argument('--max-positions', type=int, default=2)
    p.add_argument('--patterns', default='hammer,bull_engulf,doji,bull_div,flag_breakout')
    p.add_argument('--no-pattern-volume', action='store_true')
    p.add_argument('--no-bearish-exit', action='store_true')
    p.add_argument('--stop-buffer-pct', type=float, default=1.0)
    p.add_argument('--fee-bps', type=float, default=20.0)
    p.add_argument('--slippage-bps', type=float, default=5.0)
    p.add_argument('--min-notional', type=float, default=5.0)
    p.add_argument('--max-notional-pct', type=float, default=50.0)
    p.add_argument('--poll-seconds', type=int, default=12)
    p.add_argument('--universe-chunk', type=int, default=6)
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
        pattern_names=tuple([s.strip() for s in args.patterns.split(',') if s.strip()]),
        pattern_confirm_volume=(not args.no_pattern_volume),
        bearish_exit_on_pattern=(not args.no_bearish_exit),
        pattern_stop_buffer_pct=(args.stop_buffer_pct/100.0),
        fee_bps_per_side=args.fee_bps,
        slippage_bps=args.slippage_bps,
        min_notional_usd=args.min_notional,
        max_notional_pct_of_equity=(args.max_notional_pct/100.0),
    )
    ex = ExchangeClient(args.exchange, args.mode)
    if args.mode == 'backtest':
        backtest(ex, cfg)
    else:
        asyncio.run(run_loop(ex, cfg, universe=args.universe, symbols_csv=args.symbols, poll_seconds=args.poll_seconds, universe_chunk=args.universe_chunk, top_k=args.top_k))
