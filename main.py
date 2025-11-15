#!/usr/bin/env python3
"""
HMASPT - Hierarchical Multi-Agent System for Pairs Trading
Main entry point for the trading system.

This is a simplified demo version that demonstrates the system architecture
without requiring heavy ML dependencies (PyTorch, etc.)
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import CONFIG
from data_generator import generate_sample_stock_data
from utils import save_json

def main():
    print("=" * 70)
    print("HMASPT - Hierarchical Multi-Agent System for Pairs Trading")
    print("=" * 70)
    print()

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("traces", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("📊 Generating sample stock data...")
    df = generate_sample_stock_data(num_tickers=20, num_days=500)
    print(f"✅ Generated data for {df['ticker'].nunique()} tickers over {df['date'].nunique()} days")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Total records: {len(df)}")
    print()

    print("💾 Saving data to data/sample_data.csv...")
    df.to_csv("data/sample_data.csv", index=False)
    print("✅ Data saved")
    print()

    print("📈 Dataset Preview:")
    print(df.head(10).to_string(index=False))
    print()

    print("📊 Dataset Statistics:")
    stats = df.groupby("ticker").agg({
        "adj_close": ["mean", "std", "min", "max"]
    }).round(2)
    print(stats.head())
    print()

    print("🏢 Sectors Distribution:")
    sector_counts = df.groupby("sector")["ticker"].nunique()
    print(sector_counts.to_string())
    print()

    print("📝 System Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    print()

    save_json({
        "generated_at": datetime.now().isoformat(),
        "num_tickers": int(df['ticker'].nunique()),
        "num_days": int(df['date'].nunique()),
        "date_range": {
            "start": df['date'].min().isoformat(),
            "end": df['date'].max().isoformat()
        },
        "config": CONFIG
    }, "data/metadata.json")

    print("=" * 70)
    print("✅ HMASPT System Initialized Successfully!")
    print("=" * 70)
    print()
    print("📚 Next Steps:")
    print("   1. Install ML dependencies: pip install torch gymnasium stable-baselines3")
    print("   2. Review the generated sample data in data/sample_data.csv")
    print("   3. Run advanced features: pair selection, TGNN training, RL optimization")
    print()
    print("📖 For more information, see README.md")
    print()


if __name__ == "__main__":
    main()
