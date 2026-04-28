import json
import os
import logging

STATE_FILE = "bot_state.json"
logger = logging.getLogger(__name__)

def save_state(candles, wins, losses, total_pnl, regime):
    """Simpan state dengan parameter yang sudah dikonversi ke tipe native Python."""
    state = {
        'cache_candles': candles,
        'tracker_wins': int(wins),
        'tracker_losses': int(losses),
        'tracker_total_pnl_pct': float(total_pnl),
        'current_regime': int(regime)
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Gagal menyimpan state: {e}")

def load_state(cache, tracker):
    if not os.path.exists(STATE_FILE):
        return -1
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        cache.candles = state.get('cache_candles', [])
        tracker.wins = state.get('tracker_wins', 0)
        tracker.losses = state.get('tracker_losses', 0)
        tracker.total_pnl_pct = state.get('tracker_total_pnl_pct', 0.0)
        return state.get('current_regime', -1)
    except Exception as e:
        logger.error(f"Gagal memuat state: {e}")
        return -1
