from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlmodel import select

from trader.storage import new_session, Run, Trade, Log, init_db

# IMPORTANT: this imports your existing bot file by its actual name.
# If you rename the file to pattern_bot_v3.py, change the import accordingly.
import pattern_bot_v3 as pb


# --------- small helpers to write logs/trades --------------------------------

def record_log(run_id: int, level: str, message: str, context: Optional[Dict] = None) -> None:
    with new_session() as s:
        s.add(
            Log(
                run_id=run_id,
                time=datetime.now(timezone.utc),
                level=level,
                message=message,
                context=json.dumps(context or {}),
            )
        )
        s.commit()


def insert_buy(run_id: int, symbol: str, info: Dict) -> None:
    """Insert an opened trade row."""
    with new_session() as s:
        s.add(
            Trade(
                run_id=run_id,
                time=datetime.now(timezone.utc),
                symbol=symbol,
                side="long",
                qty=float(info["size"]),
                entry=float(info["entry"]),
                stop=float(info["stop"]),
                reason=f"pattern:{info.get('pattern','')}",
                meta=json.dumps(info),
            )
        )
        s.commit()


def close_latest(run_id: int, symbol: str, info: Dict) -> None:
    """Update the most recent open trade for this symbol in this run; if none, insert a closed-only row."""
    exit_price = float(info.get("exit_price", 0.0))
    pnl = float(info.get("pnl", 0.0))
    reason = info.get("exit_reason", "")
    with new_session() as s:
        row = s.exec(
            select(Trade)
            .where(Trade.run_id == run_id, Trade.symbol == symbol, Trade.exit == None)  # noqa: E711
            .order_by(Trade.time.desc())
        ).first()
        if row:
            row.exit = exit_price
            row.pnl = pnl
            row.reason = reason
            s.add(row)
        else:
            s.add(
                Trade(
                    run_id=run_id,
                    time=datetime.now(timezone.utc),
                    symbol=symbol,
                    side="close",
                    qty=0.0,
                    entry=0.0,
                    stop=0.0,
                    exit=exit_price,
                    pnl=pnl,
                    reason=reason,
                    meta=json.dumps(info),
                )
            )
        s.commit()


# --------- main worker -------------------------------------------------------

async def _pick_symbols(ex: pb.ExchangeClient, universe: Optional[str], symbols_csv: Optional[str], top_k: int) -> List[str]:
    """Use the same universe picker from your bot."""
    return await pb.pick_universe(ex, universe, symbols_csv, top_k)


async def run_worker(
    exchange_id: str = "coinbase",
    timeframe: str = "1m",
    universe: Optional[str] = None,
    symbols_csv: Optional[str] = None,
    poll_seconds: int = 12,
    universe_chunk: int = 6,
    top_k: int = 30,
    equity: float = 100.0,
    risk_pct: float = 5.0,
    max_positions: int = 2,
    stop_buffer_pct: float = 1.0,
    fee_bps: float = 20.0,
    slippage_bps: float = 5.0,
    min_notional: float = 5.0,
    max_notional_pct: float = 50.0,
    max_dday_pct: float = 3.0,
) -> None:
    """Spin the pattern bot and record BUY/SELL actions to the DB."""
    init_db()
    # New run row
    with new_session() as s:
        run = Run(mode="paper", exchange=exchange_id, timeframe=timeframe)
        s.add(run)
        s.commit()
        s.refresh(run)
        run_id = run.id

    # Build config mirroring your pattern bot
    cfg = pb.StrategyConfig(
        timeframe=timeframe,
        symbol="BTC/USD",
        equity=equity,
        risk_per_trade_pct=risk_pct / 100.0,
        max_daily_loss_pct=max_dday_pct / 100.0,
        max_positions=max_positions,
        pattern_stop_buffer_pct=stop_buffer_pct / 100.0,
        fee_bps_per_side=fee_bps,
        slippage_bps=slippage_bps,
        min_notional_usd=min_notional,
        max_notional_pct_of_equity=max_notional_pct / 100.0,
    )
    ex = pb.ExchangeClient(exchange_id, "paper")
    strat = pb.PatternStrategy(cfg)

    # Figure out the trading universe
    symbols = await _pick_symbols(ex, universe, symbols_csv, top_k)
    if not symbols:
        symbols = [cfg.symbol]

    frames: Dict[str, object] = {}
    rot = 0
    total = len(symbols)

    print(f"[worker] Starting {exchange_id.upper()} {timeframe} symbols={symbols}")
    while True:
        try:
            active = symbols if not universe_chunk or universe_chunk >= total else symbols[rot : rot + universe_chunk]
            if universe_chunk and universe_chunk < total:
                rot = (rot + universe_chunk) % total

            # Fetch OHLCV for the active batch
            for sym in active:
                raw = ex.ohlcv(sym, timeframe, limit=600)
                frames[sym] = strat.build_frame(raw)

            # Evaluate signals and record to DB
            for sym, df in frames.items():
                action, info = strat.on_new_bar(sym, df, cfg.equity)
                if action == "buy":
                    insert_buy(run_id, sym, info)
                elif action == "sell":
                    close_latest(run_id, sym, info)

            await asyncio.sleep(poll_seconds)
        except Exception as e:
            record_log(run_id, "ERROR", str(e), {"where": "loop"})
            await asyncio.sleep(30 if "429" in str(e) else 10)


# --------- CLI entrypoint ----------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--exchange", default="coinbase")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--symbols", default=None)
    p.add_argument("--universe", default=None)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--poll-seconds", type=int, default=12)
    p.add_argument("--universe-chunk", type=int, default=6)
    p.add_argument("--equity", type=float, default=100.0)
    p.add_argument("--risk", type=float, default=5.0)
    p.add_argument("--max-positions", type=int, default=2)
    p.add_argument("--stop-buffer-pct", type=float, default=1.0)
    p.add_argument("--fee-bps", type=float, default=20.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--min-notional", type=float, default=5.0)
    p.add_argument("--max-notional-pct", type=float, default=50.0)
    p.add_argument("--max-dday", type=float, default=3.0)
    args = p.parse_args()

    asyncio.run(
        run_worker(
            exchange_id=args.exchange,
            timeframe=args.timeframe,
            universe=args.universe,
            symbols_csv=args.symbols,
            top_k=args.top_k,
            poll_seconds=args.poll_seconds,
            universe_chunk=args.universe_chunk,
            equity=args.equity,
            risk_pct=args.risk,
            max_positions=args.max_positions,
            stop_buffer_pct=args.stop_buffer_pct,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            min_notional=args.min_notional,
            max_notional_pct=args.max_notional_pct,
            max_dday_pct=args.max_dday,
        )
    )
