# utils/features.py
import pandas as pd
import numpy as np
import ta

def _calculate_features_for_tf(df, tf_suffix):
    """Helper function to calculate features for a single timeframe."""
    # Basic OHLCV columns for TA-Lib
    close = df[f'c_{tf_suffix}']
    high = df[f'h_{tf_suffix}']
    low = df[f'l_{tf_suffix}']

    # --- Calculate all features based on the metadata list ---
    
    # 1. Log Return
    df[f'log_ret_1_{tf_suffix}'] = np.log(close / close.shift(1))

    # 2. EMA Slope
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df[f'ema_slope_21_{tf_suffix}'] = (ema21 - ema21.shift(1)) / ema21.shift(1)

    # 3. Price vs EMA
    ema55 = ta.trend.EMAIndicator(close, window=55).ema_indicator()
    df[f'price_vs_ema55_{tf_suffix}'] = (close - ema55) / ema55

    # 4. MACD Histogram
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df[f'macd_hist_{tf_suffix}'] = macd.macd_diff()

    # 5. ADX
    adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
    df[f'adx_{tf_suffix}'] = adx_indicator.adx()

    # 6. Normalized ATR
    atr14 = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df[f'atr_norm_{tf_suffix}'] = atr14 / close

    # 7. Bollinger Bands Width
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df[f'bb_width_{tf_suffix}'] = bb.bollinger_wband()

    # 8. Realized Volatility
    df[f'realized_vol_20_{tf_suffix}'] = df[f'log_ret_1_{tf_suffix}'].rolling(window=20).std()

    # 9. RSI
    df[f'rsi_{tf_suffix}'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    
    # 10. Rate of Change
    df[f'roc_{tf_suffix}'] = ta.momentum.ROCIndicator(close, window=12).roc()

    # 11. Stochastic Oscillator %K
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df[f'stoch_k_{tf_suffix}'] = stoch.stoch()

    # 12. Wick Ratio
    body = abs(df[f'o_{tf_suffix}'] - close)
    wick = (high - low) - body
    df[f'wick_ratio_{tf_suffix}'] = wick / (high - low)
    
    df[f'wick_ratio_{tf_suffix}'] = df[f'wick_ratio_{tf_suffix}'].fillna(0) # Handle division by zero if high==low

    return df

def generate_features(merged_df, main_tf, context_tfs):
    df = merged_df.copy()
    
    all_tfs = [main_tf] + context_tfs
    
    for tf in all_tfs:
        print(f"Generating features for timeframe: {tf}...")
        df = _calculate_features_for_tf(df, tf)
        
    # Drop rows with NaNs created by indicators' warm-up period
    # This is crucial for the scaler and model
    return df.dropna()

def merge_timeframes(main_tf_name,main_df, context_dfs):
    # Start with a copy of the main timeframe data
    # Ensure timestamp is the index and sorted for merge_asof
    main_df = main_df.copy().set_index('t').sort_index()
    main_df.columns = [f"{col}_{main_tf_name}" for col in main_df.columns]
    
    merged = main_df
    
    for tf_name, df in context_dfs.items():
        context_df = df.copy().set_index('t').sort_index()
        context_df.columns = [f"{col}_{tf_name}" for col in context_df.columns]
        
        # Use merge_asof to align the timestamps. It finds the last available
        # context candle for each main timeframe candle.
        merged = pd.merge_asof(
            left=merged,
            right=context_df,
            left_index=True,
            right_index=True,
            direction='backward' # Use previous context candle if exact match not found
        )
        
    return merged.reset_index()