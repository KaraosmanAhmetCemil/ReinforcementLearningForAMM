import random
import numpy as np
import pandas as pd


def generate_trading_data(historical_df):
    """
    Processes historical trading data for feature generation.
    :return: DataFrame with processed timestamp, price (close), and volume
    """
    df = historical_df.copy()
    df['flash_indicator'] = np.random.binomial(1, 0.2, size=len(df))  # 20% chance of flash
    # Convert timestamp from scientific notation to integer
    df['Unix Timestamp'] = df['Unix Timestamp'].astype('int64')
    # Rename and select relevant columns
    df = df.rename(columns={'Unix Timestamp': 'timestamp', 'Close': 'price', 'Volume': 'volume'})
    # Sort by timestamp and ensure proper formatting
    return df[['timestamp', 'price', 'volume', 'flash_indicator']].sort_values('timestamp').reset_index(drop=True)

def generate_alternative_data(data):
    data = data.copy()
    window_size = 10
    
    # Handle zero volumes to prevent division errors
    data['volume'] = data['volume'].replace(0, 1e-6)
    
    # Volatility Index with minimum value floor
    data['volatility'] = data['price'].rolling(window=window_size, min_periods=1).std().fillna(0.01)
    data['volatility'] = np.clip(data['volatility'], 0.01, 0.5)
    
    # Liquidity Depth calculation
    min_volume = data['volume'].quantile(0.01)
    data['liquidity_depth'] = (data['volume'] + min_volume) / (data['volatility'] + 1e-6)
    
    # Order Flow Imbalance with safeguards
    price_diff = data['price'].diff().fillna(0)
    data['order_flow_imbalance'] = price_diff / (data['volume'] + 1e-6)
    
    # Price Momentum with clipping
    data['price_momentum'] = data['price'].pct_change(periods=window_size).fillna(0)
    data['price_momentum'] = np.clip(data['price_momentum'], -0.5, 0.5)
    
    # Market Sentiment Score
    rsi = data['price'].diff().apply(lambda x: x if x > 0 else 0).rolling(window=14).mean() / \
          data['price'].diff().abs().rolling(window=14).mean()
    data['sentiment_score'] = np.clip(rsi.fillna(0.5) * 2 - 1, -1, 1)  # Added min/max values
    
    return data[['volatility', 'liquidity_depth', 'order_flow_imbalance', 'price_momentum', 'sentiment_score']].values