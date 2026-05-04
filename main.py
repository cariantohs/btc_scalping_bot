import asyncio, os, logging, json, time as _time
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

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
PORT = int(os.getenv('PORT', 8080))

bot = Bot(token=TELEGRAM_TOKEN)

# ================== MODEL ==================
MODEL_PATH = 'scalping_ensemble_v4.pkl'
SCALER_PATH = 'scaler_v4.pkl'
model = None
scaler = None

def load_models():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Ensemble model v4 dimuat")
    except Exception as e:
        logger.error(f"❌ Gagal memuat model: {e}")
    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info("✅ Scaler v4 dimuat")
    except Exception as e:
        logger.error(f"❌ Gagal memuat scaler: {e}")

# ================== CACHE ==================
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
wins = losses = 0
total_pnl = 0.0
open_trade = None
LAST_SIGNAL = None
COOLDOWN = timedelta(hours=3)   # Sinyal maksimal ~8 per hari, idealnya 1-3

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
    if model is None or scaler is None:
        logger.info("Model belum siap")
        return None

    df5 = cache.df('5m')
    df15 = cache.df('15m')
    if len(df5) < 40 or len(df15) < 30:
        logger.info("Data belum cukup")
        return None

    X = compute_features(df5)
    if X is None:
        return None

    # --- Probabilitas Ensemble ---
    X_s = scaler.transform(X)
    prob_up = model.predict_proba(X_s)[0][1]   # prob LONG
    prob_down = model.predict_proba(X_s)[0][0] # prob SHORT

    logger.info(f"Prob LONG: {prob_up:.3f} | Prob SHORT: {prob_down:.3f}")

    # Threshold sangat tinggi
    if prob_up > 0.85:
        signal = 'LONG'
    elif prob_down > 0.85:
        signal = 'SHORT'
    else:
        logger.info("Probabilitas tidak cukup tinggi")
        return None

    # --- Filter volume ---
    vol_ratio = df5['volume'].iloc[-1] / df5['volume'].rolling(20).mean().iloc[-1]
    if vol_ratio < 1.3:
        logger.info(f"Volume rendah ({vol_ratio:.2f}x rata-rata), sinyal diabaikan")
        return None

    # --- Filter ATR ---
    atr_pct = ta.volatility.AverageTrueRange(df5['high'], df5['low'], df5['close'], 14).average_true_range().iloc[-1] / df5['close'].iloc[-1] * 100
    if atr_pct < 0.3:
        logger.info(f"ATR terlalu kecil ({atr_pct:.2f}%), volatilitas rendah")
        return None

    # --- Konfirmasi tren 15m ---
    sma15 = df15['close'].rolling(20).mean().iloc[-1]
    trend15 = df15['close'].iloc[-1] > sma15
    if signal == 'LONG' and not trend15:
        logger.info("LONG ditolak: tren 15m turun")
        return None
    if signal == 'SHORT' and trend15:
        logger.info("SHORT ditolak: tren 15m naik")
        return None

    # --- Cooldown ---
    now = datetime.now()
    if LAST_SIGNAL and (now - LAST_SIGNAL) < COOLDOWN:
        logger.info("Cooldown aktif")
        return None
    if open_trade:
        logger.info("Masih ada posisi terbuka")
        return None

    # --- TP/SL ---
    cur = df5['close'].iloc[-1]
    atr_val = ta.volatility.AverageTrueRange(df5['high'], df5['low'], df5['close'], 14).average_true_range().iloc[-1]
    sl_dist = atr_val * 2.0
    if signal == 'LONG':
        sl = cur - sl_dist
        tp = cur + sl_dist * 2.0   # RR 1:2
    else:
        sl = cur + sl_dist
        tp = cur - sl_dist * 2.0

    LAST_SIGNAL = now
    logger.info(f"✅ SINYAL {signal} | Entry: {cur:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")
    return signal, tp, sl

# ================== TELEGRAM & TRADE ==================
async def send_telegram(signal, price, tp, sl):
    total = wins + losses
    wr = (wins/total*100) if total else 0
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

def update_trade(cur_price):
    global open_trade, wins, losses, total_pnl
    if not open_trade:
        return
    # Trailing stop 0.3%
    if open_trade['signal'] == 'LONG':
        if cur_price <= open_trade['entry'] * (1 - 0.3/100):
            close_trade(cur_price, 'trailing')
        elif cur_price >= open_trade['entry'] * (1 + 0.6/100):
            close_trade(cur_price, 'take_profit_hit')
    else:
        if cur_price >= open_trade['entry'] * (1 + 0.3/100):
            close_trade(cur_price, 'trailing')
        elif cur_price <= open_trade['entry'] * (1 - 0.6/100):
            close_trade(cur_price, 'take_profit_hit')
    # Time exit 30 menit
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

# ================== LISTENER ==================
async def listener():
    global open_trade
    client = None
    backoff = 1
    while True:
        try:
            client = await AsyncClient.create()
            bm = BinanceSocketManager(client)
            streams = ["btcusdt@kline_5m", "btcusdt@kline_15m", "btcusdt@kline_1h"]
            async with bm.futures_multiplex_socket(streams) as stream:
                logger.info("🔌 Terkoneksi (5m,15m,1h)")
                backoff = 1
                async def keep_alive():
                    while True:
                        await asyncio.sleep(30)
                        try:
                            if hasattr(stream, 'socket') and stream.socket:
                                await stream.socket.ping()
                        except: pass
                keep = asyncio.create_task(keep_alive())
                try:
                    while True:
                        msg = await stream.recv()
                        data = msg.get('data', {})
                        k = data.get('k', data)
                        if k and k.get('x'):
                            candle = {'timestamp': k['t'], 'open': float(k['o']), 'high': float(k['h']), 'low': float(k['l']), 'close': float(k['c']), 'volume': float(k['v'])}
                            stream_name = msg.get('stream', '')
                            if 'kline_5m' in stream_name:
                                cache.add('5m', candle)
                                cur = candle['close']
                                logger.info(f"5m close: {cur:.2f}")
                                update_trade(cur)
                                res = generate_signal()
                                if res:
                                    signal, tp, sl = res
                                    open_trade = {'signal': signal, 'entry': cur, 'time': datetime.now()}
                                    await send_telegram(signal, cur, tp, sl)
                            elif 'kline_15m' in stream_name:
                                cache.add('15m', candle)
                            elif 'kline_1h' in stream_name:
                                cache.add('1h', candle)
                        await asyncio.sleep(0.01)
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                finally:
                    keep.cancel()
        except Exception as e:
            logger.error(f"Koneksi gagal: {e}. Reconnect {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff*2, 60)
        finally:
            if client:
                await client.close_connection()

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

async def main():
    load_models()
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🚀 V4 Scalper Super Ketat aktif!")
    await asyncio.gather(start_http(), listener())

if __name__ == '__main__':
    asyncio.run(main())
