import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sample_stock_data(num_tickers=20, num_days=1000, start_date="2020-01-01"):
    """
    Generate synthetic stock price data for testing the pairs trading system.
    
    Args:
        num_tickers: Number of stock tickers to generate
        num_days: Number of trading days
        start_date: Starting date for the data
    
    Returns:
        DataFrame with columns: date, ticker, adj_close, sector
    """
    sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]
    tickers = [f"STOCK{i:03d}" for i in range(num_tickers)]
    
    dates = pd.date_range(start=start_date, periods=num_days, freq='B')
    
    records = []
    for ticker in tickers:
        sector = np.random.choice(sectors)
        
        base_price = 50 + np.random.randn() * 20
        if base_price < 10:
            base_price = 10
        
        drift = 0.0002
        volatility = 0.02
        
        prices = [base_price]
        for _ in range(num_days - 1):
            change = drift + volatility * np.random.randn()
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))
        
        for date, price in zip(dates, prices):
            records.append({
                "date": date,
                "ticker": ticker,
                "adj_close": round(price, 2),
                "sector": sector
            })
    
    df = pd.DataFrame(records)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    return df
