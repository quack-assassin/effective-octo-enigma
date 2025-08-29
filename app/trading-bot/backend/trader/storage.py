from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column
from sqlalchemy.types import DateTime
from sqlmodel import SQLModel, Field, create_engine, Session

# ---- Tables ---------------------------------------------------------------

class Run(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True)),
    )
    mode: str
    exchange: str
    timeframe: str


class Trade(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int
    time: datetime = Field(sa_column=Column(DateTime(timezone=True)))
    symbol: str
    side: str  # "long" on open, "close" when we insert a closing-only row
    qty: float
    entry: float
    stop: float
    exit: Optional[float] = None
    pnl: Optional[float] = None
    reason: Optional[str] = None
    meta: Optional[str] = None  # JSON string for extra details


class Log(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int
    time: datetime = Field(sa_column=Column(DateTime(timezone=True)))
    level: str
    message: str
    context: Optional[str] = None  # JSON


# ---- Engine & helpers -----------------------------------------------------

engine = create_engine("sqlite:///bot.db")  # swap to Postgres later if desired


def init_db() -> None:
    """Create tables if they don't exist."""
    SQLModel.metadata.create_all(engine)


def new_session() -> Session:
    """Open a SQLModel session."""
    return Session(engine)
