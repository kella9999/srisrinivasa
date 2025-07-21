import websockets
import json
import pandas as pd
import asyncio

async def fetch_crypto_data(symbol="BTCUSDT", interval="1m", csv_path="data/live.csv"):
    url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
    async with websockets.connect(url) as ws:
        while True:
            data = json.loads(await ws.recv())
            candle = data['k']
            new_row = {
                'timestamp': pd.to_datetime(candle['t'], 
                'open': float(candle['o']),
                'high': float(candle['h']),
                'low': float(candle['l']),
                'close': float(candle['c']),
                'volume': float(candle['v'])
            }
            df = pd.DataFrame([new_row])
            df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path))
