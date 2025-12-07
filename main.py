"""
HMASPT - Hierarchical Multi-Agent System for Pairs Trading
"""

import os
import sys

# Define the absolute path to the HMASPT project root
# We assume the project root is exactly where you specify it: /content/HMASPT
PROJECT_ROOT = "/content/HMASPT"

# 1. Clearer path injection for the 'src' directory
# This ensures that modules inside HMASPT/src can be imported.
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# 2. Add the project root itself to allow package-level imports (e.g., import config)
sys.path.append(PROJECT_ROOT)

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
    
    # Create project root directories (data and logs at /content/HMASPT/data, etc.)
    # Note: These will be created relative to where the script is executed.
    # To be explicit, you could use os.path.join(PROJECT_ROOT, "data")
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup environment variables
    os.environ['PYTHONHASHSEED'] = str(42)
    # The redundant sys.path.append("/content/HMASPT/") is now managed above

if __name__ == "__main__":
    main()
