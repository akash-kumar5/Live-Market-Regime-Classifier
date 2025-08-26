# utils/data_handler.py
import json
import pandas as pd
from websocket import WebSocketApp
from datetime import datetime

class BinanceCollector:
    def __init__(self, symbol="btcusdt", intervals=["5m", "15m", "1h"]):
        self.symbol = symbol.lower()
        self.intervals = intervals
        self.data = {i: pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"]) for i in intervals}

    def on_message(self, ws, message):
        msg = json.loads(message)
        
        # Check if it's a kline event in the combined stream format
        if "stream" in msg and "data" in msg and "k" in msg["data"]:
            k = msg["data"]["k"]
            interval = k["i"]
            is_closed = k["x"]

            if interval in self.intervals:
                df = self.data[interval]
                # CRITICAL FIX: Convert milliseconds to seconds by dividing by 1000
                candle_time = datetime.fromtimestamp(k["t"] / 1000)

                current_data = [
                    candle_time,
                    float(k["o"]), float(k["h"]),
                    float(k["l"]), float(k["c"]),
                    float(k["v"])
                ]

                if not df.empty and df.iloc[-1]["t"] == candle_time:
                    df.iloc[-1] = current_data
                else:
                    new_row = pd.DataFrame([current_data], columns=["t", "o", "h", "l", "c", "v"])
                    # PERFORMANCE NOTE: pd.concat is inefficient in a loop. For higher frequency,
                    # consider appending to a list and recreating the DataFrame periodically.
                    self.data[interval] = pd.concat([df, new_row], ignore_index=True)

                # LOGIC FIX: Check the length of the class attribute, not the local variable
                if len(self.data[interval]) > 1000:  # Keep the last 1000 candles
                    self.data[interval] = self.data[interval].iloc[-1000:]

    def run(self):
        streams = "/".join([f"{self.symbol}@kline_{i}" for i in self.intervals])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        ws = WebSocketApp(url, on_message=self.on_message)
        print("WebSocket connected and running.")
        ws.run_forever()