import os
import random
import numpy as np

def set_global_seed(seed: int):
    """
    Set all random seeds for reproducibility across the entire system.
    
    This sets seeds for:
    - Python's built-in random
    - NumPy
    - PyTorch (if available)
    - TensorFlow (if available)
    - Python hash seed (for dictionary ordering)
    """
    
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # TensorFlow (used by Gemini/Google AI)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"✅ Global seed set to {seed}")


# Updated CONFIG with seed and synchronized Supervisor limits
CONFIG = {
    # ===================================================================
    # GLOBAL SETTINGS
    # ===================================================================
    "random_seed": 42,  # Master seed for reproducibility
    
    # ===================================================================
    # EXISTING CONFIGURATION
    # ===================================================================
    "cointegration_pvalue_threshold": 0.05,
    "half_life_min": 1,
    "transaction_cost": 0.0005,
    "rl_policy": "MlpLstmPolicy",
    "windows": [1, 2, 4],
    "rl_timesteps": 100000,
    "half_life_max": 30,
    "rl_lookback": 30,
    "shock_prob": 0.01,
    "shock_scale": 0.4,
    "max_workers": 2,
    "initial_capital": 10000,
    "risk_free_rate": 0.00,
    
    # ===================================================================
    # SUPERVISOR RULES (Synchronized with SupervisorAgent Logic)
    # ===================================================================
    "supervisor_rules": {
        "training": {
            "min_observations": 20,
            "info_checkpoints": {
                "check_interval": 40,
                "metrics_to_track": [
                    "avg_reward",
                    "position_distribution",
                    "learning_stability"
                ]
            },
            "learning_health": {
                "reward_stability_window": 100,
                "max_reward_std_multiplier": 5.0,
                "min_exploration_rate": 0.05,
                "warn_only": True
            },
            "risk_limits": {
                "max_consecutive_losses": 30,
                "extreme_drawdown": 0.50,
                "reward_collapse": -100,
                "action": "stop_training"
            }
        },
        "holdout": {
            "min_observations": 10, # Reduced to match faster burn-in
            "check_interval": 3,    # Reduced from 30 to match Supervisor frequency
            
            # Strike 1 Warning Zone
            "info_tier": {
                "moderate_drawdown": 0.5, # Matches Strike 1 trigger
                "low_sharpe": 0.0,
                "poor_win_rate": 0.40,
                "high_volatility": 0.05,
                "action": "warn"
            },
            
            # Suggestion Zone
            "adjustment_tier": {
                "significant_drawdown": 0.1,
                "very_low_sharpe": -0.5,
                "terrible_win_rate": 0.35,
                "excessive_turnover": 50,
                "action": "suggest",
                "suggestions": {
                    "drawdown": "Consider reducing position sizes",
                    "sharpe": "Strategy may not suit current market regime",
                    "win_rate": "Entry/exit thresholds may need tightening",
                    "turnover": "Excessive trading - check for overtrading"
                }
            },
            
            # Hard Stop Zone (Strike 2 / Immediate Kill)
            "stop_tier": {
                "catastrophic_drawdown": 0.15, # Matches Hard Stop logic
                "disastrous_sharpe": -1.5,
                "consistent_failure": 0.25,
                "action": "stop"
            },
            
            # Stalemate Logic
            "position_limits": {
                "max_days_in_position": 30, # Reduced from 60 to 30
                "zero_activity_window": 30,
                "action": "info"
            },
            
            # Structural Breaks
            "anomaly_detection": {
                "reward_spike_threshold": 5.0,
                "spread_divergence": 3.0, # Matches Z-Score break
                "action": "log"
            }
        },
        "portfolio": {
            "check_frequency": 100,
            "max_portfolio_drawdown": 0.15, # Tighter portfolio limit
            "min_portfolio_sharpe": -0.3,
            "max_correlated_losses": 0.70,
            "actions": {
                "drawdown": "pause_new_pairs",
                "sharpe": "reduce_position_sizes",
                "correlation": "diversification_warning"
            }
        },
        "adaptive": {
            "volatility_adjustment": True,
            "regime_detection": {
                "low_vol": {"threshold": 0.015, "multiplier": 0.8},
                "normal_vol": {"threshold": 0.025, "multiplier": 1.0},
                "high_vol": {"threshold": 0.040, "multiplier": 1.3}
            }
        }
    }
}

# Initialize seed when module is imported
set_global_seed(CONFIG["random_seed"])
