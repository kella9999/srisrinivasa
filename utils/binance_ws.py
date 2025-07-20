import asyncio
import json
import websockets
import pandas as pd
import os
from datetime import datetime

# Folder to save data
DATA_DIR = "data"

def get_stream_name(symbol: str, interval: str) -> str:
    return f"{symbol.lower()}@kline_{interval}"

def get_csv_path(symbol: str, interval: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{symbol.upper()}_{interval}.csv")

async def stream_ohlcv(symbol="btcusdt", interval="3m"):
    stream_name = get_stream_name(symbol, interval)
    url = f"wss://stream.binance.com:9443/ws/{stream_name}"
    print(f"Connecting to {url}...")

    async with websockets.connect(url) as websocket:
        print(f"Connected to Binance WebSocket for {symbol.upper()} - {interval}")

        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                kline = data['k']

                candle = {
                    "timestamp": datetime.fromtimestamp(kline['t'] / 1000),
                    "open": float(kline['o']),
                    "high": float(kline['h']),
                    "low": float(kline['l']),
                    "close": float(kline['c']),
                    "volume": float(kline['v']),
                    "is_closed": kline['x']
                }

                if candle["is_closed"]:
                    csv_path = get_csv_path(symbol, interval)
                    df = pd.DataFrame([candle])
                    df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
                    print(f"Saved candle to {csv_path}: {candle}")

            except Exception as e:
                print(f"Error in WebSocket: {e}")
                await asyncio.sleep(5)  # Retry after 5 seconds

# Run the WebSocket (to be called by Render)
if __name__ == "__main__":
    asyncio.run(stream_ohlcv())
