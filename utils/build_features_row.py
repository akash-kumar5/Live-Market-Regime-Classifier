# utils/build_features_row.py
import pandas as pd
from utils.features import generate_features

def build_model_input(df_5m, df_15m, df_1h, last_regimes=None):
    """
    Builds a single-row DataFrame with 46 features matching the training dataset.
    """
    # Generate all technical indicators for each dataframe
    full_features_5m = generate_features(df_5m.copy())
    full_features_15m = generate_features(df_15m.copy())
    full_features_1h = generate_features(df_1h.copy())
    
    # Get just the last row of indicators
    f5 = full_features_5m.iloc[-1:].copy().drop(columns=["t"], errors="ignore")
    f15 = full_features_15m.iloc[-1:].copy().drop(columns=["t"], errors="ignore")
    f1h = full_features_1h.iloc[-1:].copy().drop(columns=["t"], errors="ignore")

    # Add suffixes
    f5 = f5.add_suffix("_5m").reset_index(drop=True)
    f15 = f15.add_suffix("_15m").reset_index(drop=True)
    f1h = f1h.add_suffix("_1h").reset_index(drop=True)

    # Get raw OHLCV data
    raw_5m = df_5m.iloc[-1][["o", "h", "l", "c", "v"]].to_frame().T.reset_index(drop=True)
    raw_5m.columns = ["open_5m", "high_5m", "low_5m", "close_5m", "volume_5m"]

    raw_15m = df_15m.iloc[-1][["o", "h", "l", "c", "v"]].to_frame().T.reset_index(drop=True)
    raw_15m.columns = ["open_15m", "high_15m", "low_15m", "close_15m", "volume_15m"]

    raw_1h = df_1h.iloc[-1][["o", "h", "l", "c", "v"]].to_frame().T.reset_index(drop=True)
    raw_1h.columns = ["open_1h", "high_1h", "low_1h", "close_1h", "volume_1h"]

    # Combine all feature groups
    features = pd.concat([
        pd.DataFrame({"timestamp": [df_5m.iloc[-1]['t']]}),
        raw_5m, raw_15m, raw_1h,
        f5, f15, f1h
    ], axis=1)

    # Add the final calculated features
    features["ATR14_5m_Mean20"] = full_features_5m["ATR14"].tail(20).mean()
    features["ATR14_15m_Mean20"] = full_features_15m["ATR14"].tail(20).mean()
    features["ATR14_1h_Mean20"] = full_features_1h["ATR14"].tail(20).mean()

    if last_regimes is None:
        last_regimes = {"5m": 0, "15m": 0, "1h": 0}
    features["Regime_5m"] = last_regimes.get("5m", 0)
    features["Regime_15m"] = last_regimes.get("15m", 0)
    features["Regime_1h"] = last_regimes.get("1h", 0)
    
    # Convert all numeric columns to float
    numeric_cols = [
        "open_5m","high_5m","low_5m","close_5m","volume_5m",
        "open_15m","high_15m","low_15m","close_15m","volume_15m",
        "open_1h","high_1h","low_1h","close_1h","volume_1h",
        "EMA20_5m","EMA50_5m","EMA200_5m","RSI14_5m","MACD_Hist_5m","ATR14_5m","BB_Width_5m","ADX14_5m",
        "EMA20_15m","EMA50_15m","EMA200_15m","RSI14_15m","MACD_Hist_15m","ATR14_15m","BB_Width_15m","ADX14_15m",
        "EMA20_1h","EMA50_1h","EMA200_1h","RSI14_1h","MACD_Hist_1h","ATR14_1h","BB_Width_1h","ADX14_1h",
        "ATR14_5m_Mean20","ATR14_15m_Mean20","ATR14_1h_Mean20"
    ]

    features[numeric_cols] = features[numeric_cols].astype(float)

    # Define the exact column order to match your training data
    final_column_order = [
        'timestamp',
        'open_5m', 'high_5m', 'low_5m', 'close_5m', 'volume_5m',
        'open_15m', 'high_15m', 'low_15m', 'close_15m', 'volume_15m',
        'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h',
        'EMA20_5m', 'EMA50_5m', 'EMA200_5m', 'RSI14_5m', 'MACD_Hist_5m', 'ATR14_5m', 'BB_Width_5m', 'ADX14_5m',
        'EMA20_15m', 'EMA50_15m', 'EMA200_15m', 'RSI14_15m', 'MACD_Hist_15m', 'ATR14_15m', 'BB_Width_15m', 'ADX14_15m',
        'EMA20_1h', 'EMA50_1h', 'EMA200_1h', 'RSI14_1h', 'MACD_Hist_1h', 'ATR14_1h', 'BB_Width_1h', 'ADX14_1h',
        'ATR14_5m_Mean20', 'Regime_5m',
        'ATR14_15m_Mean20', 'Regime_15m',
        'ATR14_1h_Mean20', 'Regime_1h'
    ]
    
    # Return the final dataframe with the correct column order
    return features[final_column_order]