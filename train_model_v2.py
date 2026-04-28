import pandas as pd
import numpy as np
import ta
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Load data futures
df = pd.read_csv('btc_futures_1m_90days.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# ================== FITUR KAYA ==================
# Returns
df['returns_1m'] = df['close'].pct_change()
df['returns_5m'] = df['close'].pct_change(5)
df['momentum_5m'] = df['close'] / df['close'].shift(5) - 1

# Indikator teknikal
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
bb = ta.volatility.BollingerBands(df['close'], window=20)
df['bb_high'] = bb.bollinger_hband()
df['bb_low'] = bb.bollinger_lband()
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

# EMA dan selisihnya
df['ema8'] = df['close'].ewm(span=8).mean()
df['ema21'] = df['close'].ewm(span=21).mean()
df['ema_diff'] = (df['ema8'] - df['ema21']) / df['close']

# Bollinger Band position (normalisasi)
df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-9)

# Volatility ratio
df['volatility'] = df['returns_1m'].rolling(20).std()

# ================== TARGET BARU (ambisius) ==================
future_returns = df['close'].shift(-5) / df['close'] - 1
df['target'] = (future_returns > 0.005).astype(int)  # naik >0.5% dalam 5 menit

# Hapus baris dengan NaN
df.dropna(inplace=True)
print(f"Data shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")

# ================== FITUR FINAL ==================
feature_cols = [
    'returns_1m', 'returns_5m', 'momentum_5m',
    'rsi', 'macd', 'macd_signal',
    'bb_high', 'bb_low', 'bb_position',
    'atr', 'volume_ratio', 'ema_diff', 'volatility'
]

X = df[feature_cols]
y = df['target']

# ================== TRAIN/TEST SPLIT (time-based) ==================
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ================== LATIH MODEL ==================
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
model.fit(X_train, y_train)

# ================== EVALUASI ==================
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(importance)

# ================== SIMPAN MODEL ==================
joblib.dump(model, 'scalping_model_v2.pkl')
print("\n✅ Model v2 disimpan: scalping_model_v2.pkl")
print("Fitur yang digunakan:", feature_cols)
print("Jumlah fitur:", len(feature_cols))