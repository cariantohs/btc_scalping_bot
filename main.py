import asyncio
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
from aiohttp import web
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager
import pandas as pd
import numpy as np
from telegram import Bot
from telegram.error import TelegramError
import ta

# Muat variabel lingkungan dari file .env (hanya untuk development lokal)
load_dotenv()

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variabel lingkungan (wajib di-set di Render)
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
PORT = int(os.getenv('PORT', 8080))  # Render menyediakan PORT

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID harus di-set di environment variables")

# Inisialisasi bot Telegram
bot = Bot(token=TELEGRAM_TOKEN)

# Cache data historis untuk perhitungan indikator
class DataCache:
    def __init__(self, maxlen=500):
        self.candles = []  # list of dict
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
        # Pastikan kolom numerik
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        return df

cache = DataCache(maxlen=200)  # Simpan 200 candle terakhir (cukup untuk indikator)

# ---------- Strategi Trading Placeholder ----------
# Ganti bagian ini dengan model ML/DL Anda sendiri
def generate_signal(df: pd.DataFrame) -> Optional[str]:
    """
    Menghasilkan sinyal trading berdasarkan indikator teknikal sederhana.
    Ini hanya contoh. Ganti dengan model Anda.
    Mengembalikan: 'LONG', 'SHORT', atau None
    """
    if len(df) < 26:  # butuh data cukup untuk MACD
        return None

    close = df['close']

    # Hitung indikator
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    macd_line = macd.macd()
    signal_line = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)

    last_idx = -1
    current_rsi = rsi.iloc[last_idx]
    current_macd = macd_line.iloc[last_idx]
    current_signal = signal_line.iloc[last_idx]
    current_bb_upper = bb.bollinger_hband().iloc[last_idx]
    current_bb_lower = bb.bollinger_lband().iloc[last_idx]
    current_price = close.iloc[last_idx]

    # Sinyal sederhana:
    # LONG jika RSI < 30 (oversold) dan harga dekat lower band
    # SHORT jika RSI > 70 (overbought) dan harga dekat upper band
    if current_rsi < 30 and current_price <= current_bb_lower * 1.01:
        return 'LONG'
    elif current_rsi > 70 and current_price >= current_bb_upper * 0.99:
        return 'SHORT'
    # Tambahan: MACD crossover (lebih kuat)
    prev_macd = macd_line.iloc[-2]
    prev_signal = signal_line.iloc[-2]
    if prev_macd < prev_signal and current_macd > current_signal:
        return 'LONG'
    elif prev_macd > prev_signal and current_macd < current_signal:
        return 'SHORT'
    return None
# ------------------------------------------------

async def send_telegram_signal(signal: str, price: float, additional_info: str = ""):
    """Mengirim sinyal ke Telegram dengan format rapi."""
    emoji = "🟢" if signal == "LONG" else "🔴"
    message = (
        f"{emoji} <b>SINYAL SCALPING BTCUSDT</b> {emoji}\n"
        f"<b>Aksi:</b> {signal}\n"
        f"<b>Harga:</b> ${price:,.2f}\n"
        f"<b>Waktu:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{additional_info}\n"
        f"<i>⚠️ Sinyal uji coba - Bukan ajakan trading</i>"
    )
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        logger.info(f"Sinyal {signal} terkirim ke Telegram")
    except TelegramError as e:
        logger.error(f"Gagal kirim Telegram: {e}")

async def handle_socket_message(msg: Dict):
    """Callback untuk setiap pesan dari WebSocket Binance."""
    if 'kline' in msg:
        k = msg['kline']
        is_closed = k['x']  # candle closed?
        if is_closed:
            candle = {
                'timestamp': k['t'],
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v']),
            }
            cache.add_candle(candle)
            logger.info(f"Candle closed: {candle['close']}")

            # Dapatkan DataFrame terbaru
            df = cache.get_dataframe()
            if df.empty:
                return

            # Generate sinyal
            signal = generate_signal(df)
            if signal:
                additional = f"RSI: {ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]:.2f}"
                await send_telegram_signal(signal, candle['close'], additional)

async def binance_websocket_listener():
    """Task untuk mendengarkan WebSocket Binance Futures."""
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)

    # Gunakan kline 1 menit untuk BTCUSDT perpetual futures
    symbol = 'BTCUSDT'
    stream_name = f"{symbol.lower()}@kline_1m"
    ts = bm.kline_futures_socket(symbol=symbol, interval='1m')

    async with ts as tscm:
        logger.info(f"Terhubung ke WebSocket Binance Futures: {stream_name}")
        # Task keep-alive: kirim ping setiap 30 detik
        async def keep_alive():
            while True:
                await asyncio.sleep(30)
                try:
                    await tscm.ping()
                    logger.debug("Ping WebSocket")
                except Exception as e:
                    logger.error(f"Gagal ping: {e}")
                    break
        keep_alive_task = asyncio.create_task(keep_alive())

        try:
            while True:
                msg = await tscm.recv()
                # Proses pesan
                await handle_socket_message(msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error di WebSocket listener: {e}")
        finally:
            keep_alive_task.cancel()
            await client.close_connection()

# ---------- HTTP Server untuk Health Check ----------
async def health_check(request):
    """Endpoint sederhana untuk memastikan service tetap hidup."""
    return web.Response(text="Bot aktif")

async def start_http_server():
    """Menjalankan server HTTP untuk health check."""
    app = web.Application()
    app.router.add_get('/', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    logger.info(f"HTTP server berjalan di port {PORT}")

# ---------- Fungsi Utama ----------
async def main():
    """Menjalankan semua komponen secara bersamaan."""
    # Kirim notifikasi startup
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"🚀 Bot Scalping BTCUSDT aktif!\nWaktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as e:
        logger.error(f"Gagal kirim notifikasi startup: {e}")

    # Jalankan HTTP server dan WebSocket listener secara concurrent
    await asyncio.gather(
        start_http_server(),
        binance_websocket_listener()
    )

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot dihentikan manual")
