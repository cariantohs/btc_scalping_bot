import asyncio
import os
import logging
import json
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, List, Tuple
import time as _time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from aiohttp import web
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager
from telegram import Bot
from telegram.error import TelegramError
import ta
import joblib
from sklearn.preprocessing import StandardScaler
from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig,
    sLSTMBlockConfig, sLSTMLayerConfig,
    FeedForwardConfig,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
PORT = int(os.getenv('PORT', 8080))

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID harus di‑set")

bot = Bot(token=TELEGRAM_TOKEN)

# ================== MODEL ==================
MODEL_XLSTM_PATH = 'xlstm_multi.pt'
SCALER_XLSTM_PATH = 'scaler_xlstm.pkl'
MODEL_LGB_PATH = 'scalping_ensemble_v4.pkl'

device = torch.device('cpu')

# ----- xLSTM -----
class MultiHorizonXSLTM(nn.Module):
    def __init__(self, input_size, hidden=128, seq_len=15):
        super().__init__()
        mlstm_config = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=4,
                num_heads=4
            )
        )
        slstm_config = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="vanilla",
                num_heads=4,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent"
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu")
        )
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mlstm_config,
            slstm_block=slstm_config,
            context_length=seq_len,
            num_blocks=2,
            embedding_dim=hidden,
            slstm_at=[1]
        )
        self.xlstm = xLSTMBlockStack(cfg)
        self.input_proj = nn.Linear(input_size, hidden)
        self.fc = nn.Linear(hidden, 3)  # 3 horizon: 5m, 15m, 30m

    def forward(self, x):
        x = self.input_proj(x)
        out = self.xlstm(x)
        last = out[:, -1, :]
        return torch.sigmoid(self.fc(last))

xlstm = None
scaler_xlstm = None
lgb = None

def load_models():
    global xlstm, scaler_xlstm, lgb
    # xLSTM
    if os.path.exists(MODEL_XLSTM_PATH) and os.path.exists(SCALER_XLSTM_PATH):
        try:
            model = MultiHorizonXSLTM(input_size=12)  # FEATURES = 12
            model.load_state_dict(torch.load(MODEL_XLSTM_PATH, map_location=device))
            model.eval()
            xlstm = model
            scaler_xlstm = joblib.load(SCALER_XLSTM_PATH)
            logger.info("✅ xLSTM & scaler dimuat")
        except Exception as e:
            logger.error(f"Gagal memuat xLSTM: {e}")
    else:
        logger.warning("⚠️ File xLSTM tidak ditemukan, hanya LightGBM yang aktif")
    
    # LightGBM
    if os.path.exists(MODEL_LGB_PATH):
        try:
            lgb = joblib.load(MODEL_LGB_PATH)
            logger.info("✅ LightGBM dimuat")
        except Exception as e:
            logger.error(f"Gagal memuat LightGBM: {e}")
    else:
        logger.warning("⚠️ LightGBM tidak ditemukan")

# ================== ORDER FLOW CACHE ==================
class OrderFlowCache:
    def __init__(self):
        self.bids = []
        self.asks = []
        self.cvd = 0.0
        self.candle_cvd = 0.0
        self.last_price = None

    def update_depth(self, bids, asks):
        self.bids = [[float(p), float(q)] for p, q in bids[:20]]
        self.asks = [[float(p), float(q)] for p, q in asks[:20]]

    def add_trade(self, price, qty, is_buyer_maker):
        delta = -qty if is_buyer_maker else qty
        self.cvd += delta
        self.candle_cvd += delta
        self.last_price = price

    def reset_candle(self):
        self.candle_cvd = 0.0

    def get_imbalance(self, depth=5):
        if not self.bids or not self.asks:
            return 0.0
        bid_vol = sum(q for _, q in self.bids[:depth])
        ask_vol = sum(q for _, q in self.asks[:depth])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

order_cache = OrderFlowCache()

# ================== CACHE MULTI TIMEFRAME ==================
class Cache:
    def __init__(self):
        self.candles_5m = deque(maxlen=100)
        self.candles_15m = deque(maxlen=60)
        self.candles_1h = deque(maxlen=30)

    def add(self, tf, candle):
        if tf == '5m':
            self.candles_5m.append(candle)
        elif tf == '15m':
            self.candles_15m.append(candle)
        elif tf == '1h':
            self.candles_1h.append(candle)

    def df(self, tf):
        data = {'5m': list(self.candles_5m), '15m': list(self.candles_15m), '1h': list(self.candles_1h)}[tf]
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        return df

cache = Cache()

# ================== PAPER TRADING ==================
wins, losses = 0, 0
total_pnl = 0.0
open_trade = None
LAST_SIGNAL = None
COOLDOWN = timedelta(hours=1.5)

# ================== FITUR ==================
FEATS = [
    'returns_1', 'returns_3', 'volatility', 'body_ratio', 'high_low_ratio',
    'rsi', 'atr_pct', 'dist_from_sma',
    'break_high', 'break_low', 'volume_ratio', 'volume_spike'
]

def compute_features(df):
    if len(df) < 30:
        return None
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']
    cur = close.iloc[-1]

    feat = {
        'returns_1': close.pct_change().iloc[-1],
        'returns_3': close.pct_change(3).iloc[-1] if len(close)>=3 else 0,
        'volatility': close.pct_change().rolling(15).std().iloc[-1],
        'body_ratio': abs(cur - df['open'].iloc[-1]) / (high.iloc[-1] - low.iloc[-1] + 1e-9),
        'high_low_ratio': (high.iloc[-1] - low.iloc[-1]) / cur,
        'rsi': ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1],
        'atr_pct': ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1] / cur * 100,
        'sma_20': close.rolling(20).mean().iloc[-1],
        'dist_from_sma': (cur - close.rolling(20).mean().iloc[-1]) / close.rolling(20).mean().iloc[-1] * 100,
        'swing_high': high.rolling(5).max().iloc[-1],
        'swing_low': low.rolling(5).min().iloc[-1],
        'break_high': int(cur > high.rolling(5).max().shift(1).iloc[-1]) if len(close)>5 else 0,
        'break_low': int(cur < low.rolling(5).min().shift(1).iloc[-1]) if len(close)>5 else 0,
        'volume_ratio': vol.iloc[-1] / vol.rolling(20).mean().iloc[-1] if vol.rolling(20).mean().iloc[-1]!=0 else 1.0,
        'volume_spike': int(vol.iloc[-1] > 1.5 * vol.rolling(20).mean().iloc[-1]) if vol.rolling(20).mean().iloc[-1]!=0 else 0
    }
    return pd.DataFrame([[feat[f] for f in FEATS]], columns=FEATS)

# ================== SINYAL ==================
def generate_signal():
    global LAST_SIGNAL, open_trade
    if lgb is None and xlstm is None:
        logger.info("Tidak ada model yang tersedia")
        return None

    df5 = cache.df('5m')
    df15 = cache.df('15m')
    if len(df5) < 30 or len(df15) < 20:
        return None

    X_static = compute_features(df5)
    if X_static is None:
        return None

    votes_long, votes_short = 0, 0
    seq_len = 15  # sesuai SEQ_LEN training

    # ----- xLSTM -----
    if xlstm and scaler_xlstm and len(cache.candles_5m) >= seq_len:
        seq_raw = []
        for c in list(cache.candles_5m)[-seq_len:]:
            seq_raw.append([c['open'], c['high'], c['low'], c['close'], c['volume']])
        seq_arr = np.array(seq_raw[-seq_len:], dtype=np.float32).reshape(1, seq_len, 5)
        # Scaling: kita perlu scaler yang dilatih untuk fitur OHLCV atau 12 fitur, tapi untuk input xLSTM kita gunakan scaler asli.
        # Sementara gunakan raw, model sudah dilatih dengan StandardScaler pada 12 fitur, jadi kita harus konversi dulu.
        # Untuk kemudahan, kita asumsikan scaler_xlstm adalah StandardScaler yang dilatih pada 12 fitur (bukan OHLCV).
        # Maka kita perlu menghitung 12 fitur untuk sekuens, bukan OHLCV.
        # Karena kompleksitas, kita gunakan X_static_scaled sebagai gantinya untuk saat ini.
        # (Akan disempurnakan setelah pelatihan final)
        pass

    # Jika xLSTM belum siap, gunakan LightGBM saja
    # ----- LightGBM -----
    if lgb:
        try:
            prob_long = lgb.predict_proba(X_static)[0][1]
            prob_short = 1 - prob_long
            if prob_long > 0.85:
                votes_long += 1
            elif prob_short > 0.85:
                votes_short += 1
        except Exception as e:
            logger.error(f"LightGBM error: {e}")

    # ----- Voting -----
    signal = None
    if votes_long >= 1 and votes_short == 0:
        signal = 'LONG'
    elif votes_short >= 1 and votes_long == 0:
        signal = 'SHORT'
    else:
        logger.info(f"Votes: L={votes_long} S={votes_short}")
        return None

    # ----- Filter Order Book -----
    imbalance = order_cache.get_imbalance()
    if signal == 'LONG' and (order_cache.candle_cvd < 0 or imbalance < 0.2):
        logger.info(f"LONG ditolak OB: CVD={order_cache.candle_cvd:.2f} Imb={imbalance:.2f}")
        return None
    if signal == 'SHORT' and (order_cache.candle_cvd > 0 or imbalance > -0.2):
        logger.info(f"SHORT ditolak OB: CVD={order_cache.candle_cvd:.2f} Imb={imbalance:.2f}")
        return None

    # ----- Filter Tren 15m -----
    sma15 = df15['close'].rolling(20).mean().iloc[-1]
    trend15_up = df15['close'].iloc[-1] > sma15
    if signal == 'LONG' and not trend15_up:
        logger.info("LONG ditolak tren 15m")
        return None
    if signal == 'SHORT' and trend15_up:
        logger.info("SHORT ditolak tren 15m")
        return None

    # ----- Cooldown -----
    now = datetime.now()
    if LAST_SIGNAL and (now - LAST_SIGNAL) < COOLDOWN:
        logger.info("Cooldown aktif")
        return None
    if open_trade:
        logger.info("Sudah ada posisi terbuka")
        return None

    # ----- TP/SL -----
    cur = df5['close'].iloc[-1]
    atr_val = ta.volatility.AverageTrueRange(df5['high'], df5['low'], df5['close'], 14).average_true_range().iloc[-1]
    sl_dist = atr_val * 2.0
    if signal == 'LONG':
        sl = cur - sl_dist
        tp = cur + sl_dist * 2.0
    else:
        sl = cur + sl_dist
        tp = cur - sl_dist * 2.0

    LAST_SIGNAL = now
    logger.info(f"✅ Sinyal {signal} | Entry: {cur:.2f} TP: {tp:.2f} SL: {sl:.2f}")
    return signal, tp, sl

# ================== TELEGRAM ==================
async def send_telegram(signal, price, tp, sl):
    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0.0
    emoji = "🟢" if signal == "LONG" else "🔴"
    msg = (
        f"{emoji} <b>SINYAL SCALPING BTCUSDT</b> {emoji}\n"
        f"<b>Aksi:</b> {signal}\n"
        f"<b>Harga Entry:</b> ${price:,.2f}\n"
        f"<b>🎯 Take Profit:</b> ${tp:,.2f}\n"
        f"<b>🛑 Stop Loss:</b> ${sl:,.2f}\n"
        f"<b>Waktu:</b> {datetime.now().strftime('%H:%M:%S')}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Total Trades: {total}\n"
        f"Win Rate: {wr:.1f}%\n"
        f"Total P/L: {total_pnl:.2f}%\n"
        f"<i>⚠️ Simulasi</i>"
    )
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='HTML')

# ================== TRADE MANAGEMENT ==================
def update_trade(cur_price):
    global open_trade, wins, losses, total_pnl
    if not open_trade:
        return
    if open_trade['signal'] == 'LONG':
        if cur_price <= open_trade['entry'] * (1 - 0.3/100):
            close_trade(cur_price, 'trailing')
        elif cur_price >= open_trade['entry'] * (1 + 0.6/100):
            close_trade(cur_price, 'take_profit')
    else:
        if cur_price >= open_trade['entry'] * (1 + 0.3/100):
            close_trade(cur_price, 'trailing')
        elif cur_price <= open_trade['entry'] * (1 - 0.6/100):
            close_trade(cur_price, 'take_profit')
    if open_trade and (datetime.now() - open_trade['time']).seconds > 1800:
        close_trade(cur_price, 'time_exit')

def close_trade(price, reason):
    global open_trade, wins, losses, total_pnl
    if not open_trade:
        return
    pnl = ((price - open_trade['entry']) / open_trade['entry'] * 100) if open_trade['signal']=='LONG' else ((open_trade['entry'] - price) / open_trade['entry'] * 100)
    if pnl > 0:
        wins += 1
    else:
        losses += 1
    total_pnl += pnl
    logger.info(f"📊 Trade closed: {open_trade['signal']} PnL {pnl:.3f}% ({reason})")
    open_trade = None

# ================== LISTENER PER STREAM (ANTI OVERFLOW) ==================
async def kline_listener():
    while True:
        client = None
        try:
            client = await AsyncClient.create()
            bm = BinanceSocketManager(client)
            async with bm.futures_socket(symbol='BTCUSDT', interval='5m') as stream:
                logger.info("🔌 Kline 5m terhubung")
                while True:
                    msg = await stream.recv()
                    # parsing...
                    await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Kline error: {e}, reconnect in 5s...")
            await asyncio.sleep(5)
        finally:
            if client:
                await client.close_connection()
            await asyncio.sleep(2)

# (mirip untuk depth, trade, 15m, 1h) ...

async def main():
    load_models()
    # tambahkan task listener dll
    # await asyncio.gather(...)
    pass

if __name__ == '__main__':
    asyncio.run(main())
