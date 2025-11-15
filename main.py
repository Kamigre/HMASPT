"""
HMASPT - Hierarchical Multi-Agent System for Pairs Trading
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("=" * 70)
    print("HMASPT - Hierarchical Multi-Agent System for Pairs Trading")
    print("=" * 70)
    print()

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("traces", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

if __name__ == "__main__":
    main()
