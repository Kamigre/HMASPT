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

    BASE_OUTPUT_DIR = "/content/drive/MyDrive/TFM test"

    required_dirs = [
        "models",
        "results",
        "reports",
        "reports/pairs",
        "traces"
    ]

    for d in required_dirs:
        full_path = os.path.join(BASE_OUTPUT_DIR, d)
        os.makedirs(full_path, exist_ok=True)
        print(f"Ensuring directory exists: {full_path}")
    
    # Create project root directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup environment variables
    os.environ['PYTHONHASHSEED'] = str(42)
    sys.path.append("/content/HMASPT/")
    
if __name__ == "__main__":
    main()
