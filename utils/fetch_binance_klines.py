# utils/fetch_binance_klines.py
import requests
import pandas as pd
from datetime import datetime
import time

def fetch_binance_klines(symbol="BTCUSDT", interval="15m", start_time_ms=None, limit=1000):
    """
    Fetches historical klines from Binance. If start_time_ms is provided, it fetches all data
    from that millisecond timestamp until now, handling pagination automatically.

    :param symbol: Trading symbol (e.g., "BTCUSDT")
    :param interval: Timeframe (e.g., "15m", "1h", "1d")
    :param start_time_ms: Start time in milliseconds since epoch.
    :return: DataFrame with kline data.
    """
    url = "https://api.binance.com/api/v3/klines"
    # limit = 1000  # Max limit per request
    all_data = []

    if start_time_ms is None:
        # For the initial fetch, get the last 1000 candles as a fallback
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(url, params=params)
        all_data = resp.json()
    else:
        # Fetch all data from start_time_ms until now
        fetch_start_time = start_time_ms
        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": fetch_start_time,
                "limit": limit
            }
            resp = requests.get(url, params=params)
            data = resp.json()
            
            if not data:
                break
                
            all_data.extend(data)
            fetch_start_time = data[-1][0] + 1
            time.sleep(0.1)

    df = pd.DataFrame(all_data, columns=[
        "t","o","h","l","c","v","close_time","q","trades","tb_base","tb_quote","ignore"
    ])
    
    if df.empty:
        return df

    df = df[["t", "o", "h", "l", "c", "v"]]
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = df[col].astype(float)
        
    df = df.drop_duplicates(subset='t', keep='last')
    return df