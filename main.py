import asyncio
import os
import logging
import json
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from aiohttp import web
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager
from telegram import Bot
from telegram.error import TelegramError
import ta
import joblib
from sklearn.preprocessing import StandardScaler

# TensorFlow hanya diimpor jika model LSTM digunakan
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
PORT = int(os.getenv('PORT', 8080))

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID harus di‑set")

bot = Bot(token=TELEGRAM_TOKEN)

# ================== MODEL ==================
MODEL_LSTM_PATH = 'scalping_lstm_entry.h5'
MODEL_LGB_PATH = 'scalping_direction_lgb.pkl'
SCALER_PATH = 'scaler_v3.pkl'

model_lstm = None
model_lgb = None
scaler = None

def load_models():
    global model_lstm, model_lgb, scaler
    if HAS_TF and os.path.exists(MODEL_LSTM_PATH):
        try:
            model_lstm = tf.keras.models.load_model(MODEL_LSTM_PATH)
            logger.info("✅ Model LSTM (entry) dimuat")
        except Exception as e:
            logger.error(f"❌ Gagal memuat LSTM: {e}")
    else:
        logger.warning("⚠️ TensorFlow tidak tersedia atau model LSTM tidak ditemukan")

    try:
        model_lgb = joblib.load(MODEL_LGB_PATH)
        logger.info("✅ Model LightGBM (direction) dimuat")
    except Exception as e:
        logger.error(f"❌ Gagal memuat LightGBM: {e}")

    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info("✅ Scaler dimuat")
    except Exception as e:
        logger.error(f"❌ Gagal memuat scaler: {e}")

# ================== STATE MANAGER ==================
STATE_FILE = 'bot_state_v3.json'

def save_state(sequence, wins, losses, total_pnl, open_trade):
    state = {
        'sequence': sequence,
        'wins': wins,
        'losses': losses,
        'total_pnl': total_pnl,
        'open_trade': open_trade
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Gagal menyimpan state: {e}")

def load_state():
    if not os.path.exists(STATE_FILE):
        return [], 0, 0, 0.0, None
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        return (state.get('sequence', []),
                state.get('wins', 0),
                state.get('losses', 0),
                state.get('total_pnl', 0.0),
                state.get('open_trade', None))
    except Exception as e:
        logger.error(f"Gagal memuat state: {e}")
        return [], 0, 0, 0.0, None

# ================== CACHE & TRACKER ==================
class MultiCache:
    def __init__(self):
        self.candles_3m = deque(maxlen=60)
        self.candles_15m = deque(maxlen=40)
        self.candles_1h = deque(maxlen=20)
        # Sequence fitur untuk LSTM (20 elemen terakhir)
        self.feature_seq = deque(maxlen=20)

    def add_candle(self, tf, candle):
        if tf == '3m':
            self.candles_3m.append(candle)
        elif tf == '15m':
            self.candles_15m.append(candle)
        elif tf == '1h':
            self.candles_1h.append(candle)

    def get_dataframe(self, tf):
        data = {
            '3m': list(self.candles_3m),
            '15m': list(self.candles_15m),
            '1h': list(self.candles_1h)
        }[tf]
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        return df

cache = MultiCache()

# Statistik & paper trading
wins = 0
losses = 0
total_pnl = 0.0
open_trade = None   # {'signal', 'entry', 'timestamp', 'bars_held'}

# Muat state sebelumnya
seq, wins, losses, total_pnl, open_trade = load_state()
cache.feature_seq = deque(seq, maxlen=20)
# ================== FITUR ==================
FEATURE_COLS = [
    'returns', 'volatility', 'body_ratio', 'high_low_ratio',
    'rsi', 'atr', 'dist_from_sma',
    'break_high', 'break_low', 'volume_ratio'
]

def compute_features(df):
    if len(df) < 30:
        return None
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    cur = close.iloc[-1]

    features = {
        'returns': close.pct_change().iloc[-1],
        'volatility': close.pct_change().rolling(20).std().iloc[-1],
        'body_ratio': abs(cur - df['open'].iloc[-1]) / (high.iloc[-1] - low.iloc[-1] + 1e-9),
        'high_low_ratio': (high.iloc[-1] - low.iloc[-1]) / cur,
        'rsi': ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1],
        'atr': ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1],
        'sma_20': close.rolling(20).mean().iloc[-1],
        'dist_from_sma': (cur - close.rolling(20).mean().iloc[-1]) / close.rolling(20).mean().iloc[-1],
        'swing_high': high.rolling(5).max().iloc[-1],
        'swing_low': low.rolling(5).min().iloc[-1],
        'break_high': int(cur > high.rolling(5).max().shift(1).iloc[-1]) if len(close) > 5 else 0,
        'break_low': int(cur < low.rolling(5).min().shift(1).iloc[-1]) if len(close) > 5 else 0,
        'volume_ratio': volume.iloc[-1] / volume.rolling(20).mean().iloc[-1] if volume.rolling(20).mean().iloc[-1] != 0 else 1.0
    }
    return pd.DataFrame([[features[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)

# ================== SINYAL ==================
def generate_signal():
    global open_trade, wins, losses, total_pnl

    if model_lgb is None or scaler is None:
        return None

    df3 = cache.get_dataframe('3m')
    df15 = cache.get_dataframe('15m')
    if len(df3) < 30 or len(df15) < 20:
        return None

    # Hitung fitur candle terbaru
    X = compute_features(df3)
    if X is None:
        return None

    # Tambahkan ke sequence (untuk LSTM)
    cache.feature_seq.append(X.iloc[0].tolist())

    # Jika model LSTM tersedia dan sequence cukup, gunakan LSTM
    if model_lstm is not None and len(cache.feature_seq) >= 20:
        # Bentuk input (1, 20, 10)
        seq_arr = np.array([list(cache.feature_seq)[-20:]])
        prob_momentum = model_lstm.predict(seq_arr, verbose=0)[0][0]
    else:
        # Fallback: gunakan LightGBM langsung untuk mendeteksi momentum (probabilitas kelas 1)
        X_scaled = scaler.transform(X)
        prob_momentum = model_lgb.predict_proba(X_scaled)[0][1] if model_lgb else 0

    logger.info(f"Prob momentum: {prob_momentum:.4f}")

    if prob_momentum < 0.8:    # threshold ketat
        return None

    # ---- Model B: penentu arah ----
    X_scaled = scaler.transform(X)
    pred_direction = model_lgb.predict(X_scaled)[0]   # 1 atau -1
    signal = 'LONG' if pred_direction == 1 else 'SHORT'

    # ---- Filter tren 15m ----
    sma_15 = df15['close'].rolling(20).mean().iloc[-1]
    trend_15_up = df15['close'].iloc[-1] > sma_15
    if signal == 'LONG' and not trend_15_up:
        logger.info("Sinyal LONG ditolak: tren 15m turun")
        return None
    if signal == 'SHORT' and trend_15_up:
        logger.info("Sinyal SHORT ditolak: tren 15m naik")
        return None

    # ---- Cooldown ----
    if open_trade is not None:
        logger.info("Masih ada trade terbuka, abaikan sinyal baru")
        return None

    # ---- TP/SL (ATR 2.5x) ----
    atr = ta.volatility.AverageTrueRange(df3['high'], df3['low'], df3['close'], 14).average_true_range().iloc[-1]
    sl_distance = atr * 2.0
    cur_price = df3['close'].iloc[-1]
    if signal == 'LONG':
        sl = cur_price - sl_distance
        tp = cur_price + sl_distance * 2.5
    else:
        sl = cur_price + sl_distance
        tp = cur_price - sl_distance * 2.5

    return signal, tp, sl

# ================== KIRIM TELEGRAM ==================
async def send_telegram_signal(signal, price, tp, sl):
    global wins, losses, total_pnl
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
        f"<i>⚠️ Simulasi – Bukan ajakan trading</i>"
    )
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='HTML')
        logger.info(f"📤 Sinyal {signal} terkirim")
    except TelegramError as e:
        logger.error(f"Gagal kirim Telegram: {e}")

# ================== PAPER TRADING ==================
def check_open_trade(current_price):
    global open_trade, wins, losses, total_pnl
    if open_trade is None:
        return

    open_trade['bars_held'] += 1
    # Exit setelah 5 candle 3m (15 menit) atau trailing stop 0.15%
    exit_bars = 5
    signal = open_trade['signal']
    entry = open_trade['entry']

    # Trailing stop 0.15% dari harga saat ini
    trailing_stop = 0.15
    if signal == 'LONG':
        if current_price <= entry * (1 - trailing_stop / 100):
            close_trade(current_price, "trailing stop")
            return
    else:
        if current_price >= entry * (1 + trailing_stop / 100):
            close_trade(current_price, "trailing stop")
            return

    if open_trade['bars_held'] >= exit_bars:
        close_trade(current_price, "time exit")

def close_trade(exit_price, reason):
    global open_trade, wins, losses, total_pnl
    pnl = ((exit_price - open_trade['entry']) / open_trade['entry'] * 100) if open_trade['signal'] == 'LONG' else ((open_trade['entry'] - exit_price) / open_trade['entry'] * 100)
    if pnl > 0:
        wins += 1
    else:
        losses += 1
    total_pnl += pnl
    logger.info(f"📊 Paper trade closed: {open_trade['signal']} PnL {pnl:.3f}% ({reason})")
    open_trade = None

# ================== LISTENER ==================
async def listener():
    global open_trade
    client = None
    while True:
        try:
            client = await AsyncClient.create()
            bm = BinanceSocketManager(client)
            streams = ["btcusdt@kline_3m", "btcusdt@kline_15m", "btcusdt@kline_1h"]
            async with bm.futures_multiplex_socket(streams) as stream:
                logger.info("🔌 Terhubung ke FUTURES streams (3m, 15m, 1h)")
                async def keep_alive():
                    while True:
                        await asyncio.sleep(30)
                        try:
                            if hasattr(stream, 'socket') and stream.socket:
                                await stream.socket.ping()
                        except: pass
                keep_alive_task = asyncio.create_task(keep_alive())

                try:
                    while True:
                        msg = await stream.recv()
                        data = msg.get('data', {})
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
                            stream_name = msg.get('stream', '')
                            if 'kline_3m' in stream_name:
                                cache.add_candle('3m', candle)
                                cur_price = candle['close']
                                # Update trailing stop & paper trading
                                if open_trade:
                                    check_open_trade(cur_price)
                                # Cek sinyal
                                res = generate_signal()
                                if res:
                                    signal, tp, sl = res
                                    # Buka trade baru
                                    open_trade = {'signal': signal, 'entry': cur_price, 'timestamp': datetime.now(), 'bars_held': 0}
                                    await send_telegram_signal(signal, cur_price, tp, sl)
                                # Simpan state
                                save_state(list(cache.feature_seq), wins, losses, total_pnl, open_trade)
                            elif 'kline_15m' in stream_name:
                                cache.add_candle('15m', candle)
                            elif 'kline_1h' in stream_name:
                                cache.add_candle('1h', candle)
                        await asyncio.sleep(0)
                except Exception as e:
                    logger.error(f"Stream inner error: {e}")
                finally:
                    keep_alive_task.cancel()
        except Exception as e:
            logger.error(f"Connection error: {e}. Reconnecting in 10s...")
            await asyncio.sleep(10)
        finally:
            if client:
                await client.close_connection()

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
    logger.info(f"🌐 Health check on port {PORT}")

# ================== MAIN ==================
async def main():
    load_models()
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🚀 V3 Scalping Bot aktif (LSTM+LightGBM, 3m/15m/1h)")
    except: pass

    await asyncio.gather(start_http(), listener())

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot dihentikan")
