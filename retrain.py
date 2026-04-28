import pandas as pd
import numpy as np
import ta
from lightgbm import LGBMClassifier
import joblib
from binance.client import Client
import os
from datetime import datetime, timedelta

SYMBOL = 'BTCUSDT'
INTERVAL = '1m'
LOOKBACK_DAYS = 30
MODEL_PATH = 'scalping_model_v2.pkl'
CSV_FILE = 'btc_futures_1m_latest.csv'

def fetch_recent_klines():
    """Ambil data 30 hari terakhir dari Binance Futures."""
    client = Client()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=LOOKBACK_DAYS)
    
    klines = client.futures_historical_klines(
        symbol=SYMBOL,
        interval=INTERVAL,
        start_str=str(int(start_time.timestamp() * 1000)),
        end_str=str(int(end_time.timestamp() * 1000)),
        limit=1000
    )
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df.set_index('timestamp', inplace=True)
    return df

def engineer_features(df):
    df['returns_1m'] = df['close'].pct_change()
    df['returns_5m'] = df['close'].pct_change(5)
    df['momentum_5m'] = df['close'] / df['close'].shift(5) - 1
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['ema8'] = df['close'].ewm(span=8).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema_diff'] = (df['ema8'] - df['ema21']) / df['close']
    df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-9)
    df['volatility'] = df['returns_1m'].rolling(20).std()
    future_returns = df['close'].shift(-5) / df['close'] - 1
    df['target'] = (future_returns > 0.005).astype(int)
    df.dropna(inplace=True)
    return df

def retrain():
    print(f"[{datetime.now()}] Memulai retraining model...")
    df = fetch_recent_klines()
    df.to_csv(CSV_FILE)
    print(f"Data {LOOKBACK_DAYS} hari terakhir disimpan: {df.shape}")
    df = engineer_features(df)
    print(f"Data setelah feature engineering: {df.shape}")
    feature_cols = [
        'returns_1m', 'returns_5m', 'momentum_5m',
        'rsi', 'macd', 'macd_signal',
        'bb_high', 'bb_low', 'bb_position',
        'atr', 'volume_ratio', 'ema_diff', 'volatility'
    ]
    X = df[feature_cols]
    y = df['target']
    model = LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model berhasil diperbarui: {MODEL_PATH}")

if __name__ == '__main__':
    retrain()