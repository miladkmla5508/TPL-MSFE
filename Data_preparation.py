import yfinance as yf
df =  yf.download('C',start='2011-10-12', end = '2024-02-01',interval='1d')
df.columns = [col[0] for col in df.columns]
df

import pandas_market_calendars as mcal
import pandas as pd

# Define NYSE calendar
nyse = mcal.get_calendar('NYSE')

# Training/testing windows (your table ranges)
windows = [
    ("2011-12-29", "2016-12-30", "2017-01-01", "2017-12-30"),
    ("2011-12-29", "2017-12-30", "2018-01-01", "2018-12-30"),
    ("2011-12-29", "2018-12-30", "2019-01-01", "2019-12-30"),
    ("2011-12-29", "2019-12-30", "2020-01-01", "2020-12-30"),
    ("2011-12-29", "2020-12-30", "2021-01-01", "2021-12-30"),
    ("2011-12-29", "2021-12-30", "2022-01-01", "2022-12-30"),
    ("2011-12-29", "2022-12-30", "2023-01-01", "2023-12-30"),
]

# Check exact trading days
results = []
for i, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
    # Training trading days
    train_days = nyse.schedule(start_date=train_start, end_date=train_end)
    train_count = len(mcal.date_range(train_days, frequency='1D'))

    # Testing trading days
    test_days = nyse.schedule(start_date=test_start, end_date=test_end)
    test_count = len(mcal.date_range(test_days, frequency='1D'))

    results.append([i, train_start, train_end, train_count, test_start, test_end, test_count])

# Convert to DataFrame for nice output
data = pd.DataFrame(results, columns=[
    "Slide", "Train Start", "Train End", "# Training Days",
    "Test Start", "Test End", "# Testing Days"
])

data


# %%
from __future__ import absolute_import
import numpy as np
import pandas as pd
import talib as tal
import ta
from pyti import catch_errors
from pyti.weighted_moving_average import weighted_moving_average as wma


def hull_moving_average(data, period):
    """
    Hull Moving Average.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    catch_errors.check_for_period_error(data, period)
    # Convert to list to avoid Series indexing issues
    data_list = data.tolist() if hasattr(data, 'tolist') else data
    hma = wma(
        2 * wma(data_list, int(period/2)) - wma(data_list, period), 
        int(np.sqrt(period))
    )
    return hma


# Define period range
periods = range(6, 21)

# =============================================================================
# PREPARE FOR BATCH COLUMN CREATION
# =============================================================================
print("Calculating technical indicators...")
new_columns = {}

# =============================================================================
# TREND INDICATORS
# =============================================================================
print("  - Trend indicators...")

# Simple Moving Average
for p in periods:
    new_columns[f'SMA_{p}'] = tal.SMA(df['Close'], timeperiod=p)

# Exponential Moving Average
for p in periods:
    new_columns[f'EMA_{p}'] = tal.EMA(df['Close'], timeperiod=p)

# Double Exponential Moving Average
for p in periods:
    new_columns[f'DEMA_{p}'] = tal.DEMA(df['Close'], timeperiod=p)

# Triple Exponential Moving Average
for p in periods:
    new_columns[f'TEMA_{p}'] = tal.TEMA(df['Close'], timeperiod=p)

# Weighted Moving Average
for p in periods:
    new_columns[f'WMA_{p}'] = tal.WMA(df['Close'], timeperiod=p)

# Triangular Moving Average
for p in periods:
    new_columns[f'TRIMA_{p}'] = tal.TRIMA(df['Close'], timeperiod=p)

# Hull Moving Average
for p in periods:
    new_columns[f'HMA_{p}'] = hull_moving_average(df['Close'], p)

# Kaufman Adaptive Moving Average
for p in periods:
    new_columns[f'KAMA_{p}'] = tal.KAMA(df['Close'], timeperiod=p)

# Volume Weighted Average Price
for p in periods:
    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=df['High'], low=df['Low'], close=df['Close'],
        volume=df['Volume'], window=p
    )
    new_columns[f'VWAP_{p}'] = vwap.volume_weighted_average_price()

# ADX - Average Directional Movement Index
for p in periods:
    adx = ta.trend.ADXIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], window=p
    )
    new_columns[f'ADX_{p}'] = adx.adx()
    new_columns[f'ADX_NEG_{p}'] = adx.adx_neg()
    new_columns[f'ADX_POS_{p}'] = adx.adx_pos()

# Aroon Indicator
for p in periods:
    aroon = ta.trend.AroonIndicator(high=df['High'], low=df['Low'], window=p)
    new_columns[f'AROON_{p}'] = aroon.aroon_indicator()
    new_columns[f'AROON_DOWN_{p}'] = aroon.aroon_down()
    new_columns[f'AROON_UP_{p}'] = aroon.aroon_up()

# Commodity Channel Index
for p in periods:
    new_columns[f'CCI_{p}'] = tal.CCI(df['High'], df['Low'], df['Close'], timeperiod=p)

# Detrended Price Oscillator
for p in periods:
    dpo = ta.trend.DPOIndicator(close=df['Close'], window=p)
    new_columns[f'DPO_{p}'] = dpo.dpo()

# MACD - Moving Average Convergence Divergence
print("  - MACD indicators...")
macd_configs = [
    (6, 20, 3), (7, 21, 4), (8, 22, 5), (9, 23, 6), (10, 24, 7),
    (11, 25, 8), (12, 26, 9), (13, 27, 10), (14, 28, 11), (15, 29, 12),
    (16, 30, 13), (17, 31, 14), (18, 32, 15), (19, 33, 16), (20, 34, 17)
]
for idx, (fast, slow, sign) in enumerate(macd_configs, start=6):
    macd = ta.trend.MACD(
        close=df['Close'], 
        window_slow=slow, 
        window_fast=fast, 
        window_sign=sign
    )
    new_columns[f'MACD_{idx}'] = macd.macd()
    new_columns[f'MACD_DIFF_{idx}'] = macd.macd_diff()
    new_columns[f'MACD_SIGNAL_{idx}'] = macd.macd_signal()

# Mass Index
mi_configs = [
    (3, 20), (4, 21), (5, 22), (6, 23), (7, 24), (8, 25), (9, 26),
    (10, 27), (11, 28), (12, 29), (13, 30), (14, 31), (15, 32), 
    (16, 33), (17, 34)
]
for idx, (fast, slow) in enumerate(mi_configs, start=6):
    mi = ta.trend.MassIndex(
        high=df['High'], low=df['Low'], 
        window_fast=fast, 
        window_slow=slow
    )
    new_columns[f'MI_{idx}'] = mi.mass_index()

# Parabolic SAR
sar_configs = [
    (0.01, 0.1), (0.01, 0.2), (0.01, 0.3), (0.01, 0.4), (0.01, 0.5),
    (0.02, 0.1), (0.02, 0.2), (0.02, 0.3), (0.02, 0.4), (0.02, 0.5),
    (0.03, 0.1), (0.03, 0.2), (0.03, 0.3), (0.03, 0.4), (0.03, 0.5)
]
for idx, (accel, maximum) in enumerate(sar_configs, start=6):
    new_columns[f'SAR_{idx}'] = tal.SAR(
        df['High'], df['Low'], 
        acceleration=accel, 
        maximum=maximum
    )

# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================
print("  - Momentum indicators...")

# Percentage Price Oscillator
ppo_configs = [
    (6, 20, 3), (7, 21, 4), (8, 22, 5), (9, 23, 6), (10, 24, 7),
    (11, 25, 8), (12, 26, 9), (13, 27, 10), (14, 28, 11), (15, 29, 12),
    (16, 30, 13), (17, 31, 14), (18, 32, 15), (19, 33, 16), (20, 34, 17)
]
for idx, (fast, slow, sign) in enumerate(ppo_configs, start=6):
    ppo = ta.momentum.PercentagePriceOscillator(
        close=df['Close'], 
        window_slow=slow, 
        window_fast=fast, 
        window_sign=sign
    )
    new_columns[f'PPO_{idx}'] = ppo.ppo()
    new_columns[f'PPO_HIST_{idx}'] = ppo.ppo_hist()
    new_columns[f'PPO_SIGNAL_{idx}'] = ppo.ppo_signal()

# Chande Momentum Oscillator
for p in periods:
    new_columns[f'CMO_{p}'] = tal.CMO(df['Close'], timeperiod=p)

# Rate of Change
for p in periods:
    new_columns[f'ROC_{p}'] = tal.ROC(df['Close'], timeperiod=p)

# Relative Strength Index
for p in periods:
    new_columns[f'RSI_{p}'] = tal.RSI(df['Close'], timeperiod=p)

# Stochastic Oscillator
stoch_configs = [
    (10, 2), (10, 3), (10, 4), (10, 5), (11, 3),
    (11, 4), (11, 5), (11, 6), (12, 4), (12, 5),
    (12, 6), (12, 7), (13, 5), (13, 6), (13, 7)
]
for idx, (window, smooth) in enumerate(stoch_configs, start=6):
    stoch = ta.momentum.StochasticOscillator(
        df['High'], df['Low'], df['Close'], 
        window=window, 
        smooth_window=smooth
    )
    new_columns[f'SLOWK_{idx}'] = stoch.stoch()
    new_columns[f'SLOWD_{idx}'] = stoch.stoch_signal()

# Ultimate Oscillator
ultosc_configs = [
    (7, 14, 26), (7, 15, 27), (7, 16, 28), (7, 17, 29), (8, 14, 26),
    (8, 15, 27), (8, 16, 28), (8, 17, 29), (9, 14, 26), (9, 15, 27),
    (9, 16, 29), (10, 14, 26), (10, 15, 27), (10, 16, 28), (10, 17, 29)
]
for idx, (t1, t2, t3) in enumerate(ultosc_configs, start=6):
    new_columns[f'ULTOSC_{idx}'] = tal.ULTOSC(
        df['High'], df['Low'], df['Close'],
        timeperiod1=t1, 
        timeperiod2=t2, 
        timeperiod3=t3
    )

# Momentum
for p in periods:
    new_columns[f'MOM_{p}'] = tal.MOM(df['Close'], timeperiod=p)

# Williams %R
for p in periods:
    new_columns[f'WILLIAM_{p}'] = tal.WILLR(
        df['High'], df['Low'], df['Close'], 
        timeperiod=p
    )

# True Strength Index
tsi_configs = [
    (15, 3), (16, 4), (17, 5), (18, 6), (19, 7),
    (20, 8), (21, 9), (22, 10), (23, 11), (24, 12),
    (25, 13), (26, 14), (27, 15), (28, 16), (29, 17)
]
for idx, (slow, fast) in enumerate(tsi_configs, start=6):
    tsi = ta.momentum.TSIIndicator(
        close=df['Close'], 
        window_slow=slow, 
        window_fast=fast
    )
    new_columns[f'TSI_{idx}'] = tsi.tsi()

# =============================================================================
# VOLUME INDICATORS
# =============================================================================
print("  - Volume indicators...")

# Accumulation/Distribution Index
ad = ta.volume.AccDistIndexIndicator(
    high=df['High'], low=df['Low'], 
    close=df['Close'], volume=df['Volume']
)
new_columns['A-D'] = ad.acc_dist_index()

# On Balance Volume
obv = ta.volume.OnBalanceVolumeIndicator(
    close=df['Close'], 
    volume=df['Volume']
)
new_columns['OBV'] = obv.on_balance_volume()

# Negative Volume Index
new_columns['NVI'] = ta.volume.negative_volume_index(
    close=df['Close'], 
    volume=df['Volume']
)

# Chaikin Money Flow
for p in periods:
    cmfi = ta.volume.ChaikinMoneyFlowIndicator(
        high=df['High'], low=df['Low'], 
        close=df['Close'], volume=df['Volume'], 
        window=p
    )
    new_columns[f'MFI_{p}'] = cmfi.chaikin_money_flow()

# Ease of Movement
for p in periods:
    eom = ta.volume.EaseOfMovementIndicator(
        high=df['High'], low=df['Low'], 
        volume=df['Volume'], 
        window=p
    )
    new_columns[f'EOM_{p}'] = eom.ease_of_movement()

# Force Index
for p in periods:
    fi = ta.volume.ForceIndexIndicator(
        close=df['Close'], 
        volume=df['Volume'], 
        window=p
    )
    new_columns[f'FI_{p}'] = fi.force_index()

# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================
print("  - Volatility indicators...")

# Average True Range
for p in periods:
    new_columns[f'ATR_{p}'] = tal.ATR(
        df['High'], df['Low'], df['Close'], 
        timeperiod=p
    )

# Bollinger Bands
bb_configs = [
    (15, 3), (16, 2), (17, 1), (18, 3), (19, 2),
    (20, 1), (21, 3), (22, 2), (23, 1), (24, 3),
    (25, 2), (26, 1), (27, 3), (28, 2), (29, 1)
]
for idx, (window, dev) in enumerate(bb_configs, start=6):
    bb = ta.volatility.BollingerBands(
        close=df['Close'], 
        window=window, 
        window_dev=dev
    )
    new_columns[f'BB_H_{idx}'] = bb.bollinger_hband()
    new_columns[f'BB_L_{idx}'] = bb.bollinger_lband()
    new_columns[f'BB_M_{idx}'] = bb.bollinger_mavg()
    new_columns[f'BB_P_{idx}'] = bb.bollinger_pband()
    new_columns[f'BB_W_{idx}'] = bb.bollinger_wband()

# Donchian Channel
for p in periods:
    new_columns[f'DC_H_{p}'] = ta.volatility.donchian_channel_hband(
        high=df['High'], low=df['Low'], close=df['Close'], 
        window=p, offset=0
    )
    new_columns[f'DC_L_{p}'] = ta.volatility.donchian_channel_lband(
        high=df['High'], low=df['Low'], close=df['Close'], 
        window=p, offset=0
    )
    new_columns[f'DC_M_{p}'] = ta.volatility.donchian_channel_mband(
        high=df['High'], low=df['Low'], close=df['Close'], 
        window=p, offset=0
    )
    new_columns[f'DC_P_{p}'] = ta.volatility.donchian_channel_pband(
        high=df['High'], low=df['Low'], close=df['Close'], 
        window=p, offset=0
    )
    new_columns[f'DC_W_{p}'] = ta.volatility.donchian_channel_wband(
        high=df['High'], low=df['Low'], close=df['Close'], 
        window=p, offset=0
    )

# Keltner Channel
kc_configs = [
    (10, 1), (10, 2), (10, 3), (10, 4), (10, 5),
    (11, 6), (12, 7), (13, 8), (14, 9), (15, 10),
    (16, 11), (17, 12), (18, 13), (19, 14), (20, 15)
]
for idx, (window, atr_window) in enumerate(kc_configs, start=6):
    kc = ta.volatility.KeltnerChannel(
        high=df['High'], low=df['Low'], close=df['Close'],
        window=window, 
        window_atr=atr_window, 
        multiplier=2
    )
    new_columns[f'KC_H_{idx}'] = kc.keltner_channel_hband()
    new_columns[f'KC_L_{idx}'] = kc.keltner_channel_lband()
    new_columns[f'KC_M_{idx}'] = kc.keltner_channel_mband()
    new_columns[f'KC_P_{idx}'] = kc.keltner_channel_pband()
    new_columns[f'KC_W_{idx}'] = kc.keltner_channel_wband()

# Ulcer Index
for p in periods:
    ui = ta.volatility.UlcerIndex(close=df['Close'], window=p)
    new_columns[f'UI_{p}'] = ui.ulcer_index()

# =============================================================================
# ADDITIONAL FEATURES
# =============================================================================
print("  - Additional features...")

# Cumulative Returns (Log Returns)
new_columns['CRet1_1'] = np.log(df['Close'] / df['Close'].shift(1))

# =============================================================================
# BATCH ADD ALL COLUMNS TO DATAFRAME
# =============================================================================
print("\nAdding all columns to dataframe...")
df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

print(f"Feature engineering complete! Total columns: {len(df.columns)}")
print(f"Total features added: {len(new_columns)}")
