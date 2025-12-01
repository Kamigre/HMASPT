CONFIG = {
    # ===================================================================
    # EXISTING CONFIGURATION
    # ===================================================================
    "cointegration_pvalue_threshold": 0.05,
    "half_life_min": 1,
    "transaction_cost": 0.0005,
    "rl_policy": "MlpPolicy",
    "windows": [1, 2, 4],
    "rl_timesteps": 50000,
    "half_life_max": 60,
    "rl_lookback": 30,
    "shock_prob": 0.01,
    "shock_scale": 0.4,
    "max_workers": 2,
    "initial_capital": 10000,
    "risk_free_rate": 0.04,  # Fixed: was 0.4 (40%!), should be 0.04 (4%)
    
    # ===================================================================
    # SUPERVISOR RULES
    # ===================================================================
    "supervisor_rules": {
        
        # ===================================================================
        # 1. TRAINING PHASE RULES (for train_on_pair)
        # ===================================================================
        "training": {
            
            # Minimum data required before making decisions
            "min_observations": 20,
            
            # --- INFORMATION CHECKPOINTS ---
            # Provide feedback without intervention
            "info_checkpoints": {
                "check_interval": 40,  # Check every N timesteps
                "metrics_to_track": [
                    "avg_reward",
                    "position_distribution",
                    "learning_stability"
                ]
            },
            
            # --- LEARNING HEALTH CHECKS ---
            "learning_health": {
                "reward_stability_window": 100,
                "max_reward_std_multiplier": 5.0,  # Flag if std > 5x mean
                "min_exploration_rate": 0.05,  # Warn if agent stops exploring
                "warn_only": True  # Don't stop, just inform
            },
            
            # --- RISK BOUNDARIES (Intervention) ---
            "risk_limits": {
                "max_consecutive_losses": 30,  # Stop if losing for 30+ steps
                "extreme_drawdown": 0.50,  # Stop at 50% drawdown
                "reward_collapse": -100,  # Stop if cumulative reward < -100
                "action": "stop_training"
            }
        },
        
        # ===================================================================
        # 2. HOLDOUT TESTING RULES (for run_operator_holdout)
        # ===================================================================
        "holdout": {
            
            # Minimum data before intervention
            "min_observations": 20,
            "check_interval": 30,  # Check every 20 steps
            
            # --- TIER 1: INFORMATIVE WARNINGS ---
            # No intervention, just log and inform
            "info_tier": {
                "moderate_drawdown": 0.15,  # 15% drawdown
                "low_sharpe": 0.0,  # Sharpe below 0
                "poor_win_rate": 0.40,  # Win rate below 40%
                "high_volatility": 0.05,  # Daily return std > 5%
                "action": "warn"
            },
            
            # --- TIER 2: ADJUSTMENT SUGGESTIONS ---
            # Suggest changes but continue trading
            "adjustment_tier": {
                "significant_drawdown": 0.25,  # 25% drawdown
                "very_low_sharpe": -0.5,  # Sharpe below -0.5
                "terrible_win_rate": 0.35,  # Win rate below 35%
                "excessive_turnover": 50,  # More than 50 trades in window
                "action": "suggest",
                "suggestions": {
                    "drawdown": "Consider reducing position sizes",
                    "sharpe": "Strategy may not suit current market regime",
                    "win_rate": "Entry/exit thresholds may need tightening",
                    "turnover": "Excessive trading - check for overtrading"
                }
            },
            
            # --- TIER 3: CRITICAL INTERVENTION ---
            # Stop trading this pair
            "stop_tier": {
                "catastrophic_drawdown": 0.40,  # 40% drawdown
                "disastrous_sharpe": -1.0,  # Sharpe below -1.0
                "consistent_failure": 0.25,  # Win rate below 25%
                "runaway_losses": -5000,  # Total P&L below -$5000
                "action": "stop"
            },
            
            # --- POSITION HEALTH CHECKS ---
            "position_limits": {
                "max_days_in_position": 60,  # Warn if stuck for 60+ days
                "zero_activity_window": 30,  # Warn if no trades for 30 steps
                "action": "info"
            },
            
            # --- STATISTICAL ANOMALIES ---
            "anomaly_detection": {
                "reward_spike_threshold": 5.0,  # |reward| > 5x recent std
                "spread_divergence": 3.0,  # Spread > 3 std from mean
                "action": "log"  # Just log, don't intervene
            }
        },
        
        # ===================================================================
        # 3. PORTFOLIO-LEVEL RULES (across all pairs)
        # ===================================================================
        "portfolio": {
            
            # Check portfolio health periodically
            "check_frequency": 100,  # Every 100 global steps
            
            # Portfolio risk limits
            "max_portfolio_drawdown": 0.30,  # 30% portfolio drawdown
            "min_portfolio_sharpe": -0.3,  # Portfolio Sharpe below -0.3
            "max_correlated_losses": 0.70,  # >70% of pairs losing simultaneously
            
            # Actions
            "actions": {
                "drawdown": "pause_new_pairs",  # Don't start new pairs
                "sharpe": "reduce_position_sizes",  # Cut all positions by 50%
                "correlation": "diversification_warning"  # Log warning
            }
        },
        
        # ===================================================================
        # 4. ADAPTIVE THRESHOLDS
        # ===================================================================
        "adaptive": {
            # Adjust thresholds based on market regime
            "volatility_adjustment": True,
            "regime_detection": {
                "low_vol": {"threshold": 0.015, "multiplier": 0.8},
                "normal_vol": {"threshold": 0.025, "multiplier": 1.0},
                "high_vol": {"threshold": 0.040, "multiplier": 1.3}
            }
        }
    }
}
