import asyncio
import pandas as pd
from binance import AsyncClient
from datetime import datetime, timedelta

async def fetch_futures_klines(symbol='BTCUSDT', interval='1m', days=90):
    client = await AsyncClient.create()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    all_klines = []
    current = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    while current < end_ts:
        klines = await client.futures_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=str(current),
            end_str=str(end_ts),
            limit=1000
        )
        if not klines:
            break
        all_klines.extend(klines)
        current = klines[-1][0] + 1
        print(f"📦 Mengambil data... total {len(all_klines)} candle")
        await asyncio.sleep(0.5)
    
    await client.close_connection()
    
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    df.to_csv('btc_futures_1m_90days.csv', index=False)
    print(f"✅ Data futures tersimpan: btc_futures_1m_90days.csv ({len(df)} baris)")

if __name__ == '__main__':
    asyncio.run(fetch_futures_klines())