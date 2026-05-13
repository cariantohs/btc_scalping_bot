import asyncio
import os
import logging
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Optional, Dict, List, Tuple

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

# ================== ZONA WAKTU WIB ==================
WIB = timezone(timedelta(hours=7))

def now_wib():
    return datetime.now(tz=WIB)

# ================== KONFIGURASI ==================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
PORT = int(os.getenv('PORT', 8080))

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID harus di‑set")

bot = Bot(token=TELEGRAM_TOKEN)

# ================== MODEL PATHS ==================
MODEL_XLSTM_PATH = 'xlstm_multi.pt'
SCALER_XLSTM_PATH = 'scaler_xlstm.pkl'
MODEL_LGB_PATH = 'scalping_ensemble_v4.pkl'

device = torch.device('cpu')

# ================== MUAT MODEL ==================
xlstm = None
scaler_xlstm = None
lgb = None

# ----- LightGBM (wajib) -----
if os.path.exists(MODEL_LGB_PATH):
    lgb = joblib.load(MODEL_LGB_PATH)
    logger.info("✅ LightGBM dimuat")
else:
    logger.error("❌ LightGBM tidak ditemukan. Bot tidak bisa berjalan.")
    exit(1)

# ----- xLSTM (opsional) -----
if os.path.exists(MODEL_XLSTM_PATH) and os.path.exists(SCALER_XLSTM_PATH):
    try:
        from xlstm import (
            xLSTMBlockStack, xLSTMBlockStackConfig,
            mLSTMBlockConfig, mLSTMLayerConfig,
            sLSTMBlockConfig, sLSTMLayerConfig,
            FeedForwardConfig,
        )
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
                self.fc = nn.Linear(hidden, 3)

            def forward(self, x):
                x = self.input_proj(x)
                out = self.xlstm(x)
                last = out[:, -1, :]
                return torch.sigmoid(self.fc(last))

        model = MultiHorizonXSLTM(input_size=12)
        model.load_state_dict(torch.load(MODEL_XLSTM_PATH, map_location=device))
        model.eval()
        xlstm = model
        scaler_xlstm = joblib.load(SCALER_XLSTM_PATH)
        logger.info("✅ xLSTM & scaler dimuat")
    except Exception as e:
        logger.warning(f"⚠️ xLSTM tidak bisa dimuat: {e}. Hanya LightGBM yang aktif.")
        xlstm = None
        scaler_xlstm = None

# ================== ORDER BOOK & CACHE ==================
class OrderFlowCache:
    def __init__(self):
        self.bids = []
        self.asks = []
        self.cvd = 0.0
        self.candle_cvd = 0.0

    def update_depth(self, bids, asks):
        self.bids = [[float(p), float(q)] for p, q in bids[:20]]
        self.asks = [[float(p), float(q)] for p, q in asks[:20]]

    def add_trade(self, price, qty, is_buyer_maker):
        delta = -qty if is_buyer_maker else qty
        self.cvd += delta
        self.candle_cvd += delta

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

class MultiCache:
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

cache = MultiCache()

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
    if lgb is None:
        logger.info("⏳ Model LightGBM belum dimuat")
        return None

    df5 = cache.df('5m')
    df15 = cache.df('15m')
    if len(df5) < 30:
        logger.info(f"⏳ Data 5m belum cukup ({len(df5)}/30)")
        return None
    if len(df15) < 20:
        logger.info(f"⏳ Data 15m belum cukup ({len(df15)}/20)")
        return None

    X_static = compute_features(df5)
    if X_static is None:
        logger.info("⏳ Gagal menghitung fitur")
        return None

    votes_long, votes_short = 0, 0

    # ----- LightGBM (threshold lebih rendah) -----
    try:
        prob_long = lgb.predict_proba(X_static)[0][1]
        prob_short = 1 - prob_long
        logger.info(f"🔍 LightGBM Prob LONG: {prob_long:.4f} | SHORT: {prob_short:.4f}")
        if prob_long > 0.70:
            votes_long += 1
            logger.info("✅ LightGBM vote LONG")
        elif prob_short > 0.70:
            votes_short += 1
            logger.info("✅ LightGBM vote SHORT")
        else:
            logger.info("❌ Probabilitas LightGBM tidak cukup tinggi (<0.70)")
    except Exception as e:
        logger.error(f"❌ LightGBM error: {e}")
        return None

    # ----- xLSTM (terintegrasi) -----
    if xlstm is not None and scaler_xlstm is not None and len(cache.candles_5m) >= 15:
        try:
            recent_df = df5.iloc[-15:]
            feat_list = []
            for i in range(len(recent_df)):
                sub_df = recent_df.iloc[:i+1]
                f = compute_features(sub_df)
                if f is not None:
                    feat_list.append(f.values[0])
                else:
                    feat_list.append(np.zeros(len(FEATS)))
            seq_arr = np.array(feat_list, dtype=np.float32)
            seq_scaled = scaler_xlstm.transform(seq_arr)
            input_tensor = torch.tensor(seq_scaled).unsqueeze(0).to(device)

            with torch.no_grad():
                xlstm_output = xlstm(input_tensor).cpu().numpy()[0]
            xlstm_long = float(xlstm_output[0])
            xlstm_short = 1.0 - xlstm_long
            logger.info(f"🧠 xLSTM Prob LONG: {xlstm_long:.4f} | SHORT: {xlstm_short:.4f}")
            if xlstm_long > 0.80:
                votes_long += 1
                logger.info("✅ xLSTM vote LONG")
            elif xlstm_short > 0.80:
                votes_short += 1
                logger.info("✅ xLSTM vote SHORT")
            else:
                logger.info("❌ xLSTM tidak cukup yakin")
        except Exception as e:
            logger.error(f"❌ xLSTM inference error: {e}")

    # ----- Voting -----
    signal = None
    if votes_long >= 1 and votes_short == 0:
        signal = 'LONG'
        logger.info("📊 Voting: LONG")
    elif votes_short >= 1 and votes_long == 0:
        signal = 'SHORT'
        logger.info("📊 Voting: SHORT")
    else:
        logger.info(f"❌ Voting tidak cukup: LONG={votes_long}, SHORT={votes_short}")
        return None

    # ----- Filter Order Book -----
    imbalance = order_cache.get_imbalance()
    cvd = order_cache.candle_cvd
    logger.info(f"📚 Order Book: CVD={cvd:.2f}, Imbalance={imbalance:.2f}")
    if signal == 'LONG':
        if cvd < 0:
            logger.info("❌ LONG ditolak: CVD negatif (tekanan jual)")
            return None
        if imbalance < 0.2:
            logger.info(f"❌ LONG ditolak: Imbalance terlalu rendah ({imbalance:.2f} < 0.2)")
            return None
    elif signal == 'SHORT':
        if cvd > 0:
            logger.info("❌ SHORT ditolak: CVD positif (tekanan beli)")
            return None
        if imbalance > -0.2:
            logger.info(f"❌ SHORT ditolak: Imbalance terlalu tinggi ({imbalance:.2f} > -0.2)")
            return None
    logger.info("✅ Filter Order Book lolos")

    # ----- Filter Tren 15m -----
    sma15 = df15['close'].rolling(20).mean().iloc[-1]
    trend15_up = df15['close'].iloc[-1] > sma15
    logger.info(f"📈 Tren 15m: {'NAIK' if trend15_up else 'TURUN'} (Close={df15['close'].iloc[-1]:.2f}, SMA20={sma15:.2f})")
    if signal == 'LONG' and not trend15_up:
        logger.info("❌ LONG ditolak: Tren 15m sedang turun")
        return None
    if signal == 'SHORT' and trend15_up:
        logger.info("❌ SHORT ditolak: Tren 15m sedang naik")
        return None
    logger.info("✅ Filter Tren 15m lolos")

    # ----- Cooldown -----
    now = now_wib()
    if LAST_SIGNAL:
        cooldown_left = COOLDOWN - (now - LAST_SIGNAL)
        if cooldown_left > timedelta(0):
            logger.info(f"⏳ Cooldown aktif ({int(cooldown_left.total_seconds()/60)} menit lagi)")
            return None
    logger.info("✅ Cooldown lolos")

    # ----- Posisi Terbuka -----
    if open_trade:
        logger.info("❌ Masih ada posisi terbuka, sinyal baru diabaikan")
        return None
    logger.info("✅ Tidak ada posisi terbuka")

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
    logger.info(f"✅ SINYAL {signal} DIKIRIM | Entry: {cur:.2f} TP: {tp:.2f} SL: {sl:.2f}")
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
        f"<b>Waktu:</b> {now_wib().strftime('%H:%M:%S')} WIB\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Total Trades: {total}\n"
        f"Win Rate: {wr:.1f}%\n"
        f"Total P/L: {total_pnl:.2f}%\n"
        f"<i>⚠️ Simulasi</i>"
    )
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='HTML')

# ================== MANAJEMEN TRADE ==================
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
    if open_trade and (now_wib() - open_trade['time']).seconds > 1800:
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

# ================== UNIFIED LISTENER (MULTIPLEX) ==================
async def unified_socket_listener():
    client = None
    backoff = 1
    while True:
        try:
            client = await AsyncClient.create()
            bm = BinanceSocketManager(client)
            streams = [
                "btcusdt@kline_5m",
                "btcusdt@kline_15m",
                "btcusdt@kline_1h",
                "btcusdt@depth20@100ms",
                "btcusdt@aggTrade"
            ]
            async with bm.futures_multiplex_socket(streams) as stream:
                if hasattr(stream, 'socket') and hasattr(stream.socket, 'max_queue'):
                    stream.socket.max_queue = 500
                logger.info("🔌 Multiplex terhubung (5m,15m,1h,depth,trade)")
                backoff = 1

                async def keep_alive():
                    while True:
                        await asyncio.sleep(30)
                        try:
                            if hasattr(stream, 'socket') and stream.socket:
                                await stream.socket.ping()
                        except:
                            pass

                keep_alive_task = asyncio.create_task(keep_alive())

                try:
                    while True:
                        msg = await stream.recv()
                        stream_name = msg.get('stream', '')
                        data = msg.get('data', {})

                        # --- Kline 5m ---
                        if 'kline_5m' in stream_name:
                            k = data.get('k', data)
                            if k and k.get('x'):
                                candle = {
                                    'timestamp': k['t'],
                                    'open': float(k['o']),
                                    'high': float(k['h']),
                                    'low': float(k['l']),
                                    'close': float(k['c']),
                                    'volume': float(k['v'])
                                }
                                cache.add('5m', candle)
                                cur = candle['close']
                                logger.info(f"5m close: {cur:.2f}")
                                order_cache.reset_candle()
                                update_trade(cur)
                                res = generate_signal()
                                if res:
                                    signal, tp, sl = res
                                    open_trade = {'signal': signal, 'entry': cur, 'time': now_wib()}
                                    await send_telegram(signal, cur, tp, sl)

                        # --- Kline 15m ---
                        elif 'kline_15m' in stream_name:
                            k = data.get('k', data)
                            if k and k.get('x'):
                                candle = {
                                    'timestamp': k['t'],
                                    'open': float(k['o']),
                                    'high': float(k['h']),
                                    'low': float(k['l']),
                                    'close': float(k['c']),
                                    'volume': float(k['v'])
                                }
                                cache.add('15m', candle)

                        # --- Kline 1h ---
                        elif 'kline_1h' in stream_name:
                            k = data.get('k', data)
                            if k and k.get('x'):
                                candle = {
                                    'timestamp': k['t'],
                                    'open': float(k['o']),
                                    'high': float(k['h']),
                                    'low': float(k['l']),
                                    'close': float(k['c']),
                                    'volume': float(k['v'])
                                }
                                cache.add('1h', candle)

                        # --- Depth ---
                        elif 'depth' in stream_name:
                            bids = data.get('bids', data.get('b', []))
                            asks = data.get('asks', data.get('a', []))
                            if bids and asks:
                                order_cache.update_depth(bids, asks)

                        # --- Trade ---
                        elif 'aggTrade' in stream_name:
                            price = float(data.get('p', 0))
                            qty = float(data.get('q', 0))
                            is_buyer_maker = data.get('m', False)
                            order_cache.add_trade(price, qty, is_buyer_maker)

                        # Jeda yang lebih besar untuk mencegah overflow
                        await asyncio.sleep(0.05)

                except Exception as e:
                    logger.error(f"Stream error: {e}")
                finally:
                    keep_alive_task.cancel()

        except Exception as e:
            logger.error(f"Koneksi gagal: {e}. Reconnect {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
        finally:
            if client:
                await client.close_connection()
            await asyncio.sleep(1)

# ================== HTTP HEALTH CHECK ==================
async def health(request):
    return web.Response(text="OK")

async def start_http():
    app = web.Application()
    app.router.add_get('/', health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    logger.info(f"🌐 Health port {PORT}")

# ================== MAIN ==================
async def main():
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🚀 Bot Scalping v4 aktif! (Multiplex, WIB)")
    except:
        pass

    await asyncio.gather(
        start_http(),
        unified_socket_listener()
    )

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot dihentikan")
