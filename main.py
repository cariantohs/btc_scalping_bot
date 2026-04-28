import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from collections import deque

from aiohttp import web
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager
import pandas as pd
import numpy as np
from telegram import Bot
from telegram.error import TelegramError
import ta
import joblib
from state_manager import save_state, load_state

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

# ---------- Model ML & HMM ----------
model = None
hmm_model = None
hmm_scaler = None
current_regime = -1
regime_names = {0: "TRENDING NAIK", 1: "TRENDING TURUN", 2: "RANGING", 3: "VOLATIL"}
last_regime_update = datetime.now()
last_signal_time: Optional[datetime] = None
SIGNAL_COOLDOWN_MINUTES = 60   # Minimal jeda antar sinyal (menit)

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
        if not bids or not asks:
            return
        try:
            self.bids = [[float(p), float(q)] for p, q in bids[:20]]
            self.asks = [[float(p), float(q)] for p, q in asks[:20]]
        except Exception as e:
            logger.error(f"Error update order book: {e}")

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

# ---------- Paper Trading ----------
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
            self.close_position(price, "forced_new_signal")
        trade = PaperTrade(signal, price, datetime.now())
        self.open_trade = trade
        logger.info(f"📊 Paper trade dibuka: {signal} @ {price:.2f}")

    def close_position(self, current_price: float, reason: str = "signal"):
        if self.open_trade is None:
            return
        trade = self.open_trade
        if trade.signal == 'LONG':
            pnl = (current_price - trade.entry_price) / trade.entry_price * 100
        else:
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
        logger.info(f"📊 Paper trade ditutup: {trade.signal} PnL: {pnl:.3f}% | Alasan: {reason}")
        self.open_trade = None

    def update_trailing_stop(self, current_price: float):
        if self.open_trade is None:
            return
        trade = self.open_trade
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

# ---------- Fungsi ATR ----------
def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return atr if pd.notna(atr) else 0.0

# ---------- Fungsi Rezim (HMM) ----------
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
    logger.info(f"🔄 Rezim pasar diperbarui: {regime_names.get(state, 'UNKNOWN')} (State {state})")

def get_current_regime_name() -> str:
    if current_regime == -1:
        return "MENGUMPULKAN DATA..."
    return regime_names.get(current_regime, "UNKNOWN")

# ---------- Strategi ML dengan Filter Ketat ----------
def generate_signal(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Menghasilkan sinyal hanya jika model ML tersedia dan sinyal sangat kuat.
    Returns:
        signal: 'LONG', 'SHORT', atau None
        take_profit: harga TP
        stop_loss: harga SL
    """
    global last_signal_time
    if model is None or len(df) < 26:
        return None, None, None

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    current_price = close.iloc[-1]

    # Hitung fitur
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
        prob_up = model.predict_proba(X)[0][1]
    except Exception as e:
        logger.error(f"Prediksi error: {e}")
        return None, None, None

    # ------ FILTER SINYAL ------
    # 1. Probabilitas sangat tinggi
    if prob_up < 0.8 and prob_up > 0.2:
        return None, None, None

    # 2. Konfirmasi tren: alignment dengan EMA 50
    ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    trend_up = current_price > ema_50
    trend_down = current_price < ema_50

    signal = None
    if prob_up > 0.8 and trend_up:
        signal = 'LONG'
    elif prob_up < 0.2 and trend_down:
        signal = 'SHORT'

    if signal is None:
        return None, None, None

    # 3. Volatilitas cukup (ATR > 0)
    atr_val = calculate_atr(df)
    if atr_val <= 0:
        return None, None, None

    # 4. Cooldown
    if last_signal_time and (datetime.now() - last_signal_time) < timedelta(minutes=SIGNAL_COOLDOWN_MINUTES):
        logger.info(f"⏳ Sinyal {signal} ditahan karena cooldown.")
        return None, None, None

    # ------ HITUNG TP/SL ------
    stop_loss_distance = atr_val * 2.0
    if signal == 'LONG':
        stop_loss = current_price - stop_loss_distance
        take_profit = current_price + (stop_loss_distance * 1.5)  # TP 1.5x SL
    else:
        stop_loss = current_price + stop_loss_distance
        take_profit = current_price - (stop_loss_distance * 1.5)

    last_signal_time = datetime.now()
    return signal, take_profit, stop_loss

def generate_signal_fallback(df: pd.DataFrame) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Fallback teknikal jika model ML tidak tersedia."""
    if len(df) < 50:
        return None, None, None

    close = df['close']
    current_price = close.iloc[-1]
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd = ta.trend.MACD(close)
    macd_line = macd.macd().iloc[-1]
    signal_line = macd.macd_signal().iloc[-1]
    bb = ta.volatility.BollingerBands(close, window=20)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

    signal = None
    if rsi < 25 and current_price <= lower * 1.005 and current_price > ema50 and macd_line > signal_line:
        signal = 'LONG'
    elif rsi > 75 and current_price >= upper * 0.995 and current_price < ema50 and macd_line < signal_line:
        signal = 'SHORT'

    if signal is None:
        return None, None, None

    atr_val = calculate_atr(df)
    if atr_val <= 0:
        return None, None, None

    stop_loss_distance = atr_val * 2.0
    if signal == 'LONG':
        stop_loss = current_price - stop_loss_distance
        take_profit = current_price + (stop_loss_distance * 1.5)
    else:
        stop_loss = current_price + stop_loss_distance
        take_profit = current_price - (stop_loss_distance * 1.5)

    return signal, take_profit, stop_loss

# ---------- Kirim Sinyal ----------
async def send_telegram_signal(signal: str, price: float, tp: float, sl: float, additional: str = ""):
    emoji = "🟢" if signal == "LONG" else "🔴"
    stats = tracker.get_stats()
    message = (
        f"{emoji} <b>SINYAL SCALPING BTCUSDT</b> {emoji}\n"
        f"<b>Aksi:</b> {signal}\n"
        f"<b>Harga Entry:</b> ${price:,.2f}\n"
        f"<b>🎯 Take Profit:</b> ${tp:,.2f}\n"
        f"<b>🛑 Stop Loss:</b> ${sl:,.2f}\n"
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
        logger.info(f"📤 Sinyal {signal} terkirim")
    except TelegramError as e:
        logger.error(f"Gagal kirim Telegram: {e}")

# ---------- Handler WebSocket ----------
async def handle_kline(data: Dict):
    k = data
    if not k:
        return
    is_closed = k.get('x', False)
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
    logger.info(f"✅ Candle 1m ditutup: {candle['close']:.2f}")

    save_state(cache, tracker, current_regime)

    micro_cache.reset_candle_cvd()
    global last_regime_update
    if datetime.now() - last_regime_update > timedelta(minutes=5):
        update_market_regime()

    df = cache.get_dataframe()
    if df.empty:
        return

    if tracker.open_trade is not None:
        trade_open_time = tracker.open_trade.timestamp
        if datetime.now() - trade_open_time >= timedelta(minutes=5):
            tracker.close_position(current_price, "time_exit_5m")

    # Cek sinyal (gunakan ML jika tersedia, jika tidak fallback)
    signal, tp, sl = generate_signal(df) if model is not None else (None, None, None)
    if signal is None:
        signal, tp, sl = generate_signal_fallback(df)

    if signal:
        tracker.open_position(signal, current_price)
        imbalance = micro_cache.get_order_book_imbalance()
        spread = micro_cache.get_spread()
        cvd = micro_cache.current_candle_cvd
        add = f"Imb: {imbalance:.2f} | Spread: {spread:.3f}% | ΔVol: {cvd:.2f}"
        await send_telegram_signal(signal, current_price, tp, sl, add)

async def handle_depth(data: Dict):
    bids = data.get('b', [])
    asks = data.get('a', [])
    if not bids or not asks:
        return
    micro_cache.update_order_book(bids, asks)

async def handle_agg_trade(data: Dict):
    price = float(data.get('p', 0))
    qty = float(data.get('q', 0))
    is_buyer_maker = data.get('m', False)
    micro_cache.add_trade(price, qty, is_buyer_maker)

# ---------- Listener per stream ----------
async def kline_listener():
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    async with bm.futures_socket(symbol='BTCUSDT', interval='1m') as stream:
        logger.info("🔌 Kline stream terhubung.")
        try:
            while True:
                msg = await stream.recv()
                if 'kline' in msg:
                    await handle_kline(msg['kline'])
                else:
                    logger.debug(f"Kline unexpected: {msg}")
        except Exception as e:
            logger.error(f"Kline socket error: {e}")
        finally:
            await client.close_connection()

async def depth_listener():
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    async with bm.futures_socket(symbol='BTCUSDT', depth='20') as stream:
        logger.info("🔌 Depth stream terhubung.")
        try:
            while True:
                msg = await stream.recv()
                if 'depth' in msg:
                    await handle_depth(msg['depth'])
                else:
                    logger.debug(f"Depth unexpected: {msg}")
        except Exception as e:
            logger.error(f"Depth socket error: {e}")
        finally:
            await client.close_connection()

async def trade_listener():
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    async with bm.futures_socket(symbol='BTCUSDT') as stream:
        logger.info("🔌 Trade stream terhubung.")
        try:
            while True:
                msg = await stream.recv()
                if 'aggTrade' in msg:
                    await handle_agg_trade(msg['aggTrade'])
                else:
                    logger.debug(f"Trade unexpected: {msg}")
        except Exception as e:
            logger.error(f"Trade socket error: {e}")
        finally:
            await client.close_connection()

# ---------- HTTP Server ----------
async def health_check(request):
    return web.Response(text="OK")

async def start_http_server():
    app = web.Application()
    app.router.add_get('/', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    logger.info(f"🌐 HTTP server port {PORT}")

# ---------- Muat Model ----------
def load_models():
    global model, hmm_model, hmm_scaler
    try:
        model = joblib.load('scalping_model_v1.pkl')
        logger.info("✅ Model ML v1 berhasil dimuat")
    except Exception as e:
        logger.error(f"❌ Gagal memuat model ML: {e}")
        model = None
    try:
        hmm_model = joblib.load('hmm_model.pkl')
        hmm_scaler = joblib.load('hmm_scaler.pkl')
        logger.info("✅ Model HMM berhasil dimuat")
    except Exception as e:
        logger.error(f"❌ Gagal memuat HMM: {e}")
        hmm_model = None
        hmm_scaler = None

# ---------- Main ----------
async def main():
    load_models()
    global current_regime
    saved_regime = load_state(cache, tracker)
    if saved_regime != -1:
        current_regime = saved_regime
        logger.info(f"📥 State dipulihkan. Rezim: {get_current_regime_name()}, Total Trades: {tracker.wins + tracker.losses}")
    else:
        logger.info("🆕 Tidak ada state tersimpan. Memulai dari awal.")

    http_server_task = asyncio.create_task(start_http_server())
    await asyncio.sleep(1)

    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"🚀 Bot Scalping FUTURES aktif!\nRezim: {get_current_regime_name()}\nMode: ML + TP/SL + Filter Ketat"
        )
    except Exception as e:
        logger.error(f"Gagal kirim notifikasi startup: {e}")

    await asyncio.gather(
        kline_listener(),
        depth_listener(),
        trade_listener()
    )
    http_server_task.cancel()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot dihentikan manual")
