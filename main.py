import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from collections import deque

import aiohttp
from aiohttp import web
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager
import pandas as pd
import numpy as np
from telegram import Bot
from telegram.error import TelegramError
import ta
import joblib

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
    raise ValueError("TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID harus di-set")

bot = Bot(token=TELEGRAM_TOKEN)

# ---------- Muat Model ML ----------
model = None
try:
    model = joblib.load('scalping_model_v1.pkl')
    logger.info("✅ Model ML v1 dimuat")
except FileNotFoundError:
    logger.warning("⚠️ Model ML tidak ditemukan, fallback ke strategi sederhana")
except Exception as e:
    logger.error(f"❌ Gagal memuat model ML: {e}")

# ---------- Muat HMM ----------
hmm_model = None
hmm_scaler = None
try:
    hmm_model = joblib.load('hmm_model.pkl')
    hmm_scaler = joblib.load('hmm_scaler.pkl')
    logger.info("✅ Model HMM untuk deteksi rezim dimuat")
except FileNotFoundError:
    logger.warning("⚠️ Model HMM tidak ditemukan, deteksi rezim dinonaktifkan")
except Exception as e:
    logger.error(f"❌ Gagal memuat HMM: {e}")

current_regime = -1
regime_names = {0: "TRENDING NAIK", 1: "TRENDING TURUN", 2: "RANGING", 3: "VOLATIL"}
last_regime_update = datetime.now()

# ---------- Cache Mikrostruktur ----------
class MicrostructureCache:
    def __init__(self):
        self.bids: List[List[float]] = []
        self.asks: List[List[float]] = []
        self.last_update_id: Optional[int] = None
        self.trade_history: deque = deque(maxlen=500)
        self.cvd: float = 0.0
        self.current_candle_cvd: float = 0.0

    def update_order_book(self, bids: List[List[str]], asks: List[List[str]]):
        self.bids = [[float(p), float(q)] for p, q in bids]
        self.asks = [[float(p), float(q)] for p, q in asks]

    def add_trade(self, price: float, quantity: float, is_buyer_maker: bool):
        delta = -quantity if is_buyer_maker else quantity
        self.cvd += delta
        self.current_candle_cvd += delta
        self.trade_history.append({
            'price': price,
            'quantity': quantity,
            'delta': delta,
            'timestamp': datetime.now()
        })

    def reset_candle_cvd(self):
        self.current_candle_cvd = 0.0

    def get_order_book_imbalance(self, depth: int = 5) -> float:
        if not self.bids or not self.asks:
            return 0.0
        total_bid = sum(q for _, q in self.bids[:depth])
        total_ask = sum(q for _, q in self.asks[:depth])
        if total_bid + total_ask == 0:
            return 0.0
        return (total_bid - total_ask) / (total_bid + total_ask)

    def get_spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        return (best_ask - best_bid) / best_ask * 100

    def get_volume_delta(self) -> float:
        return sum(t['delta'] for t in self.trade_history)

micro_cache = MicrostructureCache()

# ---------- Cache OHLCV ----------
class DataCache:
    def __init__(self, maxlen=500):
        self.candles = []
        self.maxlen = maxlen

    def add_candle(self, candle: Dict):
        self.candles.append(candle)
        if len(self.candles) > self.maxlen:
            self.candles.pop(0)

    def get_dataframe(self) -> pd.DataFrame:
        if not self.candles:
            return pd.DataFrame()
        df = pd.DataFrame(self.candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        return df

cache = DataCache(maxlen=200)

# ---------- Paper Trading Simulator ----------
class PaperTrade:
    def __init__(self, signal: str, entry_price: float, timestamp: datetime):
        self.signal = signal
        self.entry_price = entry_price
        self.timestamp = timestamp
        self.closed = False
        self.exit_price = None
        self.pnl_pct = 0.0
        self.exit_reason = ""

class PerformanceTracker:
    def __init__(self):
        self.trades: List[PaperTrade] = []
        self.wins = 0
        self.losses = 0
        self.total_pnl_pct = 0.0
        self.open_trade: Optional[PaperTrade] = None

    def open_position(self, signal: str, price: float):
        if self.open_trade is not None:
            # Paksa tutup posisi sebelumnya jika ada (harusnya tidak terjadi)
            self.close_position(price, "forced_new_signal")

        trade = PaperTrade(signal, price, datetime.now())
        self.open_trade = trade
        logger.info(f"📊 Paper trade opened: {signal} @ {price:.2f}")

    def close_position(self, current_price: float, reason: str = "signal"):
        if self.open_trade is None:
            return

        trade = self.open_trade
        if trade.signal == 'LONG':
            pnl = (current_price - trade.entry_price) / trade.entry_price * 100
        else:  # SHORT
            pnl = (trade.entry_price - current_price) / trade.entry_price * 100

        trade.exit_price = current_price
        trade.pnl_pct = pnl
        trade.exit_reason = reason
        trade.closed = True

        self.trades.append(trade)
        self.total_pnl_pct += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        logger.info(f"📊 Paper trade closed: {trade.signal} PnL: {pnl:.3f}% | Reason: {reason}")
        self.open_trade = None

    def update_trailing_stop(self, current_price: float):
        """Cek apakah posisi harus ditutup karena trailing stop."""
        if self.open_trade is None:
            return

        trade = self.open_trade
        # Trailing stop 0.15% dari harga entry
        trailing_pct = 0.15
        if trade.signal == 'LONG':
            stop_price = trade.entry_price * (1 - trailing_pct / 100)
            if current_price <= stop_price:
                self.close_position(current_price, "trailing_stop")
        else:
            stop_price = trade.entry_price * (1 + trailing_pct / 100)
            if current_price >= stop_price:
                self.close_position(current_price, "trailing_stop")

    def get_stats(self) -> Dict:
        total_trades = self.wins + self.losses
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0.0
        avg_pnl = self.total_pnl_pct / total_trades if total_trades > 0 else 0.0
        return {
            'total_trades': total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'total_pnl_pct': self.total_pnl_pct,
            'avg_pnl': avg_pnl,
            'open_position': self.open_trade is not None
        }

tracker = PerformanceTracker()

# ---------- Fungsi Update Rezim ----------
def update_market_regime():
    global current_regime, last_regime_update
    if hmm_model is None or hmm_scaler is None:
        return

    df = cache.get_dataframe()
    if len(df) < 30:
        return

    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(20)
    df['momentum'] = df['close'] / df['close'].shift(20) - 1
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df.dropna(inplace=True)

    if df.empty:
        return

    latest = df.iloc[-1:][['volatility', 'momentum', 'volume_ratio']].values
    latest_scaled = hmm_scaler.transform(latest)
    state = hmm_model.predict(latest_scaled)[0]
    current_regime = state
    last_regime_update = datetime.now()
    logger.info(f"🔄 Rezim pasar: {regime_names.get(state, 'UNKNOWN')}")

def get_current_regime_name() -> str:
    return regime_names.get(current_regime, "UNKNOWN")

# ---------- Strategi ----------
def generate_signal(df: pd.DataFrame) -> Optional[str]:
    selected_model = model
    if selected_model is None or len(df) < 26:
        return generate_signal_fallback(df)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    features = {}
    features['returns_1m'] = close.pct_change().iloc[-1]
    features['returns_5m'] = close.pct_change(5).iloc[-1] if len(close) >= 6 else 0
    features['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd = ta.trend.MACD(close)
    features['macd'] = macd.macd().iloc[-1]
    features['macd_signal'] = macd.macd_signal().iloc[-1]
    bb = ta.volatility.BollingerBands(close, window=20)
    features['bb_high'] = bb.bollinger_hband().iloc[-1]
    features['bb_low'] = bb.bollinger_lband().iloc[-1]
    features['atr'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    vol_ma = volume.rolling(20).mean().iloc[-1]
    features['volume_ratio'] = volume.iloc[-1] / vol_ma if vol_ma != 0 else 1.0

    feature_order = [
        'returns_1m', 'returns_5m', 'rsi', 'macd', 'macd_signal',
        'bb_high', 'bb_low', 'atr', 'volume_ratio'
    ]
    X = pd.DataFrame([features])[feature_order]

    try:
        prob_up = selected_model.predict_proba(X)[0][1]
    except Exception as e:
        logger.error(f"Prediksi error: {e}")
        return None

    if prob_up > 0.65:
        return 'LONG'
    elif prob_up < 0.35:
        return 'SHORT'
    return None

def generate_signal_fallback(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 26:
        return None
    close = df['close']
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd = ta.trend.MACD(close)
    macd_line = macd.macd().iloc[-1]
    signal_line = macd.macd_signal().iloc[-1]
    bb = ta.volatility.BollingerBands(close, window=20)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    price = close.iloc[-1]

    if rsi < 30 and price <= lower * 1.01:
        return 'LONG'
    elif rsi > 70 and price >= upper * 0.99:
        return 'SHORT'
    prev_macd = macd.macd().iloc[-2]
    prev_signal = macd.macd_signal().iloc[-2]
    if prev_macd < prev_signal and macd_line > signal_line:
        return 'LONG'
    elif prev_macd > prev_signal and macd_line < signal_line:
        return 'SHORT'
    return None

# ---------- Kirim Sinyal + Statistik ----------
async def send_telegram_signal(signal: str, price: float, additional: str = ""):
    emoji = "🟢" if signal == "LONG" else "🔴"
    stats = tracker.get_stats()
    message = (
        f"{emoji} <b>SINYAL SCALPING BTCUSDT</b> {emoji}\n"
        f"<b>Aksi:</b> {signal}\n"
        f"<b>Harga:</b> ${price:,.2f}\n"
        f"<b>Rezim:</b> {get_current_regime_name()}\n"
        f"<b>Waktu:</b> {datetime.now().strftime('%H:%M:%S')}\n"
        f"{additional}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<b>📊 PERFORMANCE (Paper Trading)</b>\n"
        f"Total Trades: {stats['total_trades']}\n"
        f"Win Rate: {stats['win_rate']:.1f}% ({stats['wins']}W / {stats['losses']}L)\n"
        f"Total Gain: {stats['total_pnl_pct']:.2f}%\n"
        f"Avg P/L: {stats['avg_pnl']:.3f}%\n"
        f"<i>⚠️ Simulasi - Bukan trading riil</i>"
    )
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        logger.info(f"📤 Sinyal {signal} + statistik terkirim")
    except TelegramError as e:
        logger.error(f"Gagal kirim: {e}")

# ---------- Handler WebSocket ----------
async def handle_kline(data: Dict):
    k = data.get('k', data)
    if not k:
        return
    is_closed = k.get('x', False)

    # Selalu update harga terbaru untuk trailing stop (baik closed maupun belum)
    current_price = float(k.get('c', 0))
    if tracker.open_trade is not None:
        tracker.update_trailing_stop(current_price)

    if not is_closed:
        return

    candle = {
        'timestamp': k['t'],
        'open': float(k['o']),
        'high': float(k['h']),
        'low': float(k['l']),
        'close': current_price,
        'volume': float(k['v']),
    }
    cache.add_candle(candle)
    logger.info(f"✅ Candle closed: {candle['close']:.2f}")

    micro_cache.reset_candle_cvd()

    global last_regime_update
    if datetime.now() - last_regime_update > timedelta(minutes=5):
        update_market_regime()

    df = cache.get_dataframe()
    if df.empty:
        return

    # Jika ada posisi terbuka dan candle kelipatan 5 (5 menit), tutup paksa (time-based exit)
    if tracker.open_trade is not None:
        trade_open_time = tracker.open_trade.timestamp
        if datetime.now() - trade_open_time >= timedelta(minutes=5):
            tracker.close_position(current_price, "time_exit_5m")

    signal = generate_signal(df)
    if signal:
        # Buka posisi baru (akan otomatis menutup posisi lama jika ada)
        tracker.open_position(signal, current_price)

        imbalance = micro_cache.get_order_book_imbalance()
        spread = micro_cache.get_spread()
        cvd = micro_cache.current_candle_cvd
        add = f"Imb: {imbalance:.2f} | Spread: {spread:.3f}% | ΔVol: {cvd:.2f}"
        await send_telegram_signal(signal, current_price, add)

async def handle_depth(data: Dict):
    bids = data.get('b', [])
    asks = data.get('a', [])
    if bids and asks:
        micro_cache.update_order_book(bids, asks)

async def handle_agg_trade(data: Dict):
    price = float(data.get('p', 0))
    qty = float(data.get('q', 0))
    is_buyer_maker = data.get('m', False)
    micro_cache.add_trade(price, qty, is_buyer_maker)

# ---------- Listener Utama ----------
async def unified_socket_listener():
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)

    streams = [
        "btcusdt@kline_1m",
        "btcusdt@depth20@100ms",
        "btcusdt@aggTrade"
    ]

    try:
        async with bm.futures_multiplex_socket(streams) as stream:
            logger.info("🔌 Terhubung ke multiple FUTURES streams (Fase 3 + Tracker)")

            async def keep_alive():
                while True:
                    await asyncio.sleep(30)
                    try:
                        if hasattr(stream, 'socket') and stream.socket:
                            await stream.socket.ping()
                    except Exception:
                        pass

            keep_alive_task = asyncio.create_task(keep_alive())

            try:
                while True:
                    msg = await stream.recv()
                    stream_name = msg.get('stream', '')
                    data = msg.get('data', {})

                    if 'kline' in stream_name:
                        await handle_kline(data)
                    elif 'depth20' in stream_name:
                        await handle_depth(data)
                    elif 'aggTrade' in stream_name:
                        await handle_agg_trade(data)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Socket error: {e}")
            finally:
                keep_alive_task.cancel()
    except Exception as e:
        logger.error(f"Multiplex connection error: {e}")
    finally:
        await client.close_connection()

# ---------- HTTP Server ----------
async def health_check(request):
    return web.Response(text="Bot aktif")

async def start_http_server():
    app = web.Application()
    app.router.add_get('/', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    logger.info(f"🌐 HTTP server port {PORT}")

# ---------- Main ----------
async def main():
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"🚀 Bot Scalping FUTURES aktif!\nRezim: {get_current_regime_name()}\nMode: Paper Trading + Statistik"
        )
    except Exception as e:
        logger.error(f"Startup notify error: {e}")

    await asyncio.gather(
        start_http_server(),
        unified_socket_listener()
    )

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped")