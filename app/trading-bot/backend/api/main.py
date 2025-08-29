from fastapi import FastAPI
from sqlmodel import select
from trader.storage import init_db, new_session, Trade, Log

app = FastAPI(title="Trading Bot API")
init_db()

@app.get("/trades")
def list_trades(limit: int = 100):
    with new_session() as s:
        rows = s.exec(select(Trade).order_by(Trade.time.desc()).limit(limit)).all()
        return rows

@app.get("/logs")
def list_logs(limit: int = 200):
    with new_session() as s:
        rows = s.exec(select(Log).order_by(Log.time.desc()).limit(limit)).all()
        return rows
