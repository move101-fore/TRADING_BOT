from fastapi import FastAPI
from backend import config

app = FastAPI(title="Trading Bot")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": config.EXECUTION_MODE,
        "primary_timeframe": config.PRIMARY_TIMEFRAME,
    }
