# HMASPT
## Hierarchical Multi-Agent System for Pairs Trading

## 🎯 Overview

This system implements a multi-agent architecture for identifying and trading cointegrated stock pairs. It combines:

- **Temporal Graph Neural Networks** to learn dynamic relationships between stocks.
- **Reinforcement Learning** for optimal trading policy.
- **Multi-agent coordination** for distributed decision-making.

## 🚀 Quick Start

To run manually:
```bash
python main.py
```

## 📊 Features

### Current Features
- ✅ Statistical utilities (half-life, spread calculation)
- ✅ Configuration system
- ✅ Data persistence and logging
- ✅ Temporal Graph Neural Network for pair selection
- ✅ Reinforcement Learning trading environment
- ✅ Multi-agent coordination system

## 📁 Project Structure

```
.
├── main.py                    # Main entry point
├── src/
│   ├── config.py             # System configuration
│   ├── utils.py              # Statistical utilities
│   └── agents/               # Multi-agent system
│       ├── __init__.py
│       ├── message_bus.py    # MessageBus, JSONLogger, Graph
│       ├── selector_agent.py # TGNN pair selection
│       ├── operator_agent.py # RL trading execution
│       └── supervisor_agent.py # Portfolio monitoring
├── data/                     # Generated data files
├── models/                   # Model checkpoints
├── traces/                   # Event traces
├── logs/                     # System logs
├── requirements.txt          # Python dependencies
└── Agents_13112025.ipynb     # Original notebook (reference)
```

## 🔧 Configuration

Edit `src/config.py` to adjust system parameters:

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

## 📦 Dependencies

### Installed (Core)
- numpy, pandas - Data manipulation
- scikit-learn - Machine learning utilities
- statsmodels - Statistical modeling
- scipy - Scientific computing
- matplotlib - Visualization
- tqdm - Progress bars

### Optional (Advanced Features)
To enable full ML functionality:
```bash
pip install torch torch-geometric gymnasium stable-baselines3 faiss-cpu
```

## 🏗️ Architecture

The system uses a hierarchical multi-agent design:

1. **SelectorAgent** - Identifies cointegrated pairs using TGNN
2. **OperatorAgent** - Executes trades using RL-optimized strategies
3. **SupervisorAgent** - Coordinates agents and monitors system health
4. **MessageBus** - Facilitates inter-agent communication

## 📖 Documentation

- `replit.md` - Project overview and setup
- `AGENTS.md` - Detailed agent system documentation
- `Agents_13112025.ipynb` - Original Jupyter notebook (reference)

## ⚠️ Disclaimer

This is a research/educational project. Not intended for live trading without extensive testing and risk management.
