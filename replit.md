# HMASPT - Hierarchical Multi-Agent System for Pairs Trading

## Overview

This project implements a Hierarchical Multi-Agent System for Pairs Trading, originally developed in Jupyter Notebook format. The system uses advanced machine learning techniques including:

- **Temporal Graph Neural Networks (TGNN)** for pair selection
- **Reinforcement Learning (RL)** for trading strategy optimization
- **Multi-agent architecture** for coordinated trading decisions

## Project Status

**Current State**: Simplified demo version with working core functionality

The project has been converted from a Jupyter notebook to a standalone Python application. The current version includes:
- ✅ Sample stock data generation
- ✅ Configuration system
- ✅ Utility functions for statistical analysis
- ⏳ TGNN implementation (requires PyTorch)
- ⏳ RL trading environment (requires Gymnasium & Stable-Baselines3)

## Project Structure

```
.
├── main.py                 # Main entry point
├── src/
│   ├── config.py          # System configuration
│   ├── utils.py           # Utility functions (half-life, spread calculation)
│   └── data_generator.py  # Sample data generation
├── data/                  # Generated data files
├── models/                # Saved model checkpoints
├── traces/                # Event traces for debugging
├── logs/                  # System logs
├── requirements.txt       # Python dependencies
└── Agents_13112025.ipynb  # Original notebook (reference)
```

## Recent Changes

**2025-11-15**: Initial import from GitHub
- Converted Jupyter notebook to Python modules
- Created sample data generator
- Set up basic project structure
- Installed core dependencies (numpy, pandas, scikit-learn, etc.)

## User Preferences

- **Development Style**: Data science / Machine Learning focused
- **Tech Stack**: Python, PyTorch, Graph Neural Networks, Reinforcement Learning

## Architecture

The system uses a multi-agent architecture:

1. **SelectorAgent**: Uses TGNN to identify cointegrated stock pairs
2. **OperatorAgent**: Executes trades based on RL-optimized strategies
3. **SupervisorAgent**: Coordinates agents and monitors system health
4. **MessageBus**: Facilitates inter-agent communication

## Configuration

System parameters are defined in `src/config.py`:

```python
CONFIG = {
    "cointegration_pvalue_threshold": 0.05,
    "half_life_min": 1,
    "transaction_cost": 0.0005,
    "rl_policy": "MlpPolicy",
    "windows": [60],
    "rl_timesteps": 50000,
    "half_life_max": 60
}
```

## Running the System

The demo version runs automatically in the Replit environment. To run manually:

```bash
python main.py
```

This generates sample stock data and demonstrates the system initialization.

## Next Steps

To enable full functionality:

1. Install ML dependencies:
   ```bash
   pip install torch gymnasium stable-baselines3 torch-geometric
   ```

2. Implement remaining agent modules from the original notebook

3. Add real market data integration

## Dependencies

### Core (Installed)
- numpy
- pandas
- scikit-learn
- statsmodels
- scipy
- matplotlib
- tqdm

### ML (Optional)
- torch
- torch-geometric
- gymnasium
- stable-baselines3
- faiss-cpu

## Notes

- This is a research/educational project for pairs trading strategies
- The original implementation was developed for Google Colab
- Sample data is generated synthetically for demonstration
- Full implementation requires significant computational resources for training TGNN and RL models
