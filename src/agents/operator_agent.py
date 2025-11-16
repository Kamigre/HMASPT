"""
Enhanced Operator Agent for pairs trading with advanced RL features.
Requires: gymnasium, stable-baselines3 (optional)
"""

import os
import json
import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    print("Warning: gymnasium not installed. OperatorAgent will have limited functionality.")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. RL training will not be available.")

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    PARALLEL_TRAINING_AVAILABLE = True
except ImportError:
    PARALLEL_TRAINING_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import CONFIG
from utils import half_life, compute_spread
from agents.message_bus import MessageBus, JSONLogger

if GYMNASIUM_AVAILABLE:
    class PairTradingEnv(gym.Env):
        """
        Advanced Gymnasium environment for pairs trading.
        
        Improvements:
        - Dynamic position sizing (0-100% allocation)
        - Stop-loss mechanism
        - Enhanced reward shaping with baseline
        - Market regime awareness
        - Better risk-adjusted metrics
        """
        
        metadata = {"render.modes": ["human", "plot"]}

        def __init__(self, series_x: pd.Series, series_y: pd.Series, lookback: int = 500,
                    shock_prob: float = 0.01, shock_scale: float = 0.1,
                    initial_capital: float = 1000, test_mode: bool = False,
                    stop_loss_threshold: float = 0.15, max_position_hold: int = 60,
                    baseline_sharpe: float = 0.0):
            super().__init__()

            self.align = pd.concat([series_x, series_y], axis=1).dropna()
            self.lookback = lookback
            self.test_mode = test_mode
            
            # Always start after lookback period
            self.start_idx = lookback
            self.ptr = self.start_idx

            # IMPROVEMENT 1: Continuous action space for dynamic position sizing
            # Action: [position_direction (-1 to 1), position_size (0 to 1)]
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), 
                                          high=np.array([1.0, 1.0]), 
                                          dtype=np.float32)

            self.position = 0.0  # Continuous position: -1 to 1
            self.position_size = 0.0  # Size: 0 to 1
            self.initial_capital = initial_capital
            self.portfolio_value = initial_capital
            self.cash = initial_capital
            self.cum_returns = 0.0
            self.peak = initial_capital
            self.max_drawdown = 0.0
            self.trades = []
            
            # Position tracking
            self.shares_x = 0.0
            self.shares_y = 0.0
            self.position_entry_ptr = None
            self.position_hold_duration = 0
            
            # IMPROVEMENT 2: Stop-loss and risk management
            self.stop_loss_threshold = stop_loss_threshold
            self.max_position_hold = max_position_hold
            self.stop_loss_triggered = False
            
            # IMPROVEMENT 3: Baseline for reward shaping
            self.baseline_sharpe = baseline_sharpe

            # No shocks in test mode
            self.shock_prob = shock_prob if not test_mode else 0.0  
            self.shock_scale = shock_scale if not test_mode else 0.0

            from utils import compute_spread, half_life
            self.spread = compute_spread(self.align.iloc[:, 0], self.align.iloc[:, 1])
            n = len(self.spread)
            shock_mask = np.random.rand(n) < self.shock_prob
            self.shocks = np.random.randn(n) * self.shock_scale * self.spread.std() * shock_mask
            self.spread_shocked = self.spread + self.shocks

            # Enhanced feature computation
            self.zscores = (self.spread_shocked - self.spread_shocked.rolling(self.lookback).mean()) / \
                          (self.spread_shocked.rolling(self.lookback).std() + 1e-8)
            self.vols = self.spread_shocked.rolling(15).std()
            self.rx = self.align.iloc[:, 0].pct_change()
            self.ry = self.align.iloc[:, 1].pct_change()
            self.corrs = self.rx.rolling(15).corr(self.ry)
            
            # IMPROVEMENT 4: Market regime indicators
            # Volatility regime
            self.vol_mean = self.vols.rolling(15).mean()
            self.vol_regime = self.vols / (self.vol_mean + 1e-8)
            
            # Momentum indicators
            self.momentum_x = self.align.iloc[:, 0].pct_change(15)
            self.momentum_y = self.align.iloc[:, 1].pct_change(15)

            self.zscores_np = np.nan_to_num(self.zscores.to_numpy())
            self.vols_np = np.nan_to_num(self.vols.to_numpy())
            self.corrs_np = np.nan_to_num(self.corrs.to_numpy())
            self.vol_regime_np = np.nan_to_num(self.vol_regime.to_numpy())
            self.spread_np = self.spread_shocked.to_numpy()
            
            self.prices_x = self.align.iloc[:, 0].to_numpy()
            self.prices_y = self.align.iloc[:, 1].to_numpy()

        def _compute_features(self, idx: int):
            """Compute enhanced observation features."""
            from utils import half_life
            from config import CONFIG
            
            z = self.zscores_np[idx]
            vol = self.vols_np[idx]
            corr = self.corrs_np[idx]
            vol_regime = self.vol_regime_np[idx]
            
            # Half-life for mean reversion speed
            start = max(0, idx - self.lookback)
            hl = half_life(self.spread_np[start:idx+1]) if idx > start else CONFIG["half_life_max"]
            
            # Current position info
            position_info = self.position * self.position_size
            
            return np.array([z, vol, hl, corr, vol_regime, position_info], dtype=np.float32)

        def _calculate_position_value(self, idx: int):
            """Calculate current position value."""
            if abs(self.position * self.position_size) < 1e-6:
                return 0.0
            
            position_value_x = self.shares_x * self.prices_x[idx]
            position_value_y = self.shares_y * self.prices_y[idx]
            return position_value_x + position_value_y

        def _check_stop_loss(self, idx: int):
            """Check if stop-loss should be triggered."""
            if self.position_entry_ptr is None:
                return False
            
            # Calculate current position P&L
            current_value = self._calculate_position_value(idx)
            entry_value = self.position_entry_value if hasattr(self, 'position_entry_value') else 0
            
            if abs(entry_value) > 1e-6:
                pnl_pct = (current_value - entry_value) / abs(entry_value)
                if pnl_pct < -self.stop_loss_threshold:
                    return True
            
            # Check maximum hold duration
            if self.position_hold_duration >= self.max_position_hold:
                return True
            
            return False

        def _execute_trade(self, old_position: float, old_size: float, 
                          new_position: float, new_size: float, idx: int):
            """
            Execute trade with dynamic position sizing.
            
            Position interpretation:
            - position > 0: Long spread (Buy X, Sell Y)
            - position < 0: Short spread (Sell X, Buy Y)
            - position = 0 or size = 0: Close all positions
            """
            from config import CONFIG
            
            transaction_cost = 0.0
            
            # Close old position if it exists
            if abs(old_position * old_size) > 1e-6:
                self.cash += self.shares_x * self.prices_x[idx]
                self.cash += self.shares_y * self.prices_y[idx]
                
                transaction_cost += CONFIG["transaction_cost"] * 2
                
                self.shares_x = 0.0
                self.shares_y = 0.0
                self.position_entry_ptr = None
                self.position_hold_duration = 0
            
            # Open new position if not neutral
            if abs(new_position * new_size) > 1e-6:
                # Dynamic allocation based on position_size
                allocation = self.portfolio_value * 0.5 * new_size
                
                if new_position > 0:  # Long spread
                    self.shares_x = allocation / self.prices_x[idx]
                    self.shares_y = -allocation / self.prices_y[idx]
                else:  # Short spread
                    self.shares_x = -allocation / self.prices_x[idx]
                    self.shares_y = allocation / self.prices_y[idx]
                
                self.cash -= self.shares_x * self.prices_x[idx]
                self.cash -= self.shares_y * self.prices_y[idx]
                
                transaction_cost += CONFIG["transaction_cost"] * 2
                
                # Track position entry
                self.position_entry_ptr = idx
                self.position_entry_value = allocation * 2
                self.position_hold_duration = 0
            
            return transaction_cost

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            """Reset environment to initial state."""
            super().reset(seed=seed)
            
            self.ptr = self.start_idx
            
            self.position = 0.0
            self.position_size = 0.0
            self.portfolio_value = self.initial_capital
            self.cash = self.initial_capital
            self.cum_returns = 0.0
            self.peak = self.initial_capital
            self.max_drawdown = 0.0
            self.trades = []
            self.shares_x = 0.0
            self.shares_y = 0.0
            self.position_entry_ptr = None
            self.position_hold_duration = 0
            self.stop_loss_triggered = False
            
            obs = self._compute_features(self.ptr)
            return obs, {}

        def step(self, action: np.ndarray):
            """Execute one step with continuous actions."""
            from config import CONFIG
            
            # Parse continuous actions
            target_position = np.clip(action[0], -1.0, 1.0)
            target_size = np.clip(action[1], 0.0, 1.0)
            
            # Calculate current portfolio value
            position_value = self._calculate_position_value(self.ptr)
            self.portfolio_value = self.cash + position_value
            
            # Check stop-loss
            force_close = self._check_stop_loss(self.ptr)
            if force_close:
                target_position = 0.0
                target_size = 0.0
                self.stop_loss_triggered = True
            
            # Execute trade if position changes significantly
            transaction_cost = 0.0
            position_changed = (abs(target_position - self.position) > 0.1 or 
                              abs(target_size - self.position_size) > 0.1)
            
            if position_changed or force_close:
                transaction_cost = self._execute_trade(
                    self.position, self.position_size,
                    target_position, target_size, self.ptr
                )
                self.position = target_position
                self.position_size = target_size
            else:
                # Update hold duration
                if self.position_entry_ptr is not None:
                    self.position_hold_duration += 1
            
            # Move to next time step
            next_ptr = self.ptr + 1
            terminated = next_ptr >= len(self.spread_np)
            truncated = False

            if terminated:
                final_position_value = self._calculate_position_value(self.ptr)
                self.portfolio_value = self.cash + final_position_value
                
                obs_next = self._compute_features(self.ptr)
                return obs_next, 0.0, terminated, truncated, {
                    "cum_reward": self.cum_returns,
                    "max_drawdown": self.max_drawdown,
                    "position": self.position,
                    "final_value": self.portfolio_value
                }

            # Calculate new portfolio value after price changes
            new_position_value = self._calculate_position_value(next_ptr)
            new_portfolio_value = self.cash + new_position_value - transaction_cost
            
            # IMPROVEMENT 5: Enhanced reward shaping
            step_return = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
            
            # Penalize excessive risk
            risk_penalty = 0.0
            if self.max_drawdown > 0.20:  # Penalize if drawdown > 20%
                risk_penalty = -0.001 * (self.max_drawdown - 0.20)
            
            # Reward for mean reversion success
            reversion_bonus = 0.0
            if self.position_entry_ptr is not None and abs(self.position * self.position_size) > 1e-6:
                z_at_entry = self.zscores_np[self.position_entry_ptr]
                z_current = self.zscores_np[next_ptr]
                if abs(z_current) < abs(z_at_entry):  # Spread converging
                    reversion_bonus = 0.0005
            
            # Stop-loss penalty
            sl_penalty = -0.01 if self.stop_loss_triggered else 0.0
            
            # Combined reward with baseline shaping
            reward = step_return + risk_penalty + reversion_bonus + sl_penalty
            
            # Update portfolio value
            self.portfolio_value = new_portfolio_value
            self.cum_returns = (self.portfolio_value / self.initial_capital - 1) * 100
            
            # Update peak and drawdown
            self.peak = max(self.peak, self.portfolio_value)
            current_drawdown = (self.peak - self.portfolio_value) / self.peak
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            self.ptr = next_ptr
            self.trades.append(step_return)
            self.stop_loss_triggered = False
            
            obs_next = self._compute_features(self.ptr)
            
            return obs_next, float(reward), terminated, truncated, {
                "pnl": step_return,
                "cum_reward": self.cum_returns,
                "max_drawdown": self.max_drawdown,
                "position": self.position * self.position_size,
                "portfolio_value": self.portfolio_value
            }

if SB3_AVAILABLE:
    class PerformanceCallback(BaseCallback):
        """Callback to track training performance."""
        
        def __init__(self, check_freq: int = 1000, verbose: int = 0):
            super().__init__(verbose)
            self.check_freq = check_freq
            self.episode_rewards = []
            self.episode_lengths = []
            
        def _on_step(self) -> bool:
            if self.n_calls % self.check_freq == 0:
                if len(self.episode_rewards) > 0:
                    mean_reward = np.mean(self.episode_rewards[-10:])
                    if self.verbose > 0:
                        print(f"Steps: {self.n_calls}, Mean Reward: {mean_reward:.4f}")
            return True
        
        def _on_rollout_end(self) -> None:
            pass

@dataclass
class OperatorAgent:
    """
    Enhanced agent for RL-based pairs trading execution.
    
    New Features:
    - Dynamic position sizing
    - Stop-loss management
    - Enhanced reward shaping
    - Multiple RL algorithms support
    - Better performance tracking
    """
    
    message_bus: MessageBus = None
    logger: JSONLogger = None
    storage_dir: str = "models/"

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        self.active = True
        self.transaction_cost = CONFIG["transaction_cost"]
        
        # Tracking for supervisor monitoring
        self.current_step = 0
        self.traces_buffer = []
        self.max_buffer_size = 10000
        
        # Performance history
        self.training_history = []

    def get_current_step(self):
        """Return current step count."""
        return self.current_step

    def get_traces_since_step(self, start_step):
        """Get traces since specific step."""
        return [t for t in self.traces_buffer if t.get('step', 0) >= start_step]

    def add_trace(self, trace):
        """Add trace to buffer."""
        self.traces_buffer.append(trace)
        
        if len(self.traces_buffer) > self.max_buffer_size:
            self.traces_buffer = self.traces_buffer[-self.max_buffer_size:]

    def clear_traces_before_step(self, step):
        """Clear old traces."""
        self.traces_buffer = [t for t in self.traces_buffer if t.get('step', 0) >= step]

    def apply_command(self, command):
        """Apply runtime commands."""
        cmd_type = command.get("command")
        if cmd_type == "adjust_transaction_cost":
            old = self.transaction_cost
            self.transaction_cost = command.get("new_value", old)
            CONFIG["transaction_cost"] = self.transaction_cost
            self.logger.log("operator", "adjust_transaction_cost", {
                "old_value": old, "new_value": self.transaction_cost
            })
        elif cmd_type == "pause":
            self.active = False
            self.logger.log("operator", "paused", {})
        elif cmd_type == "resume":
            self.active = True
            self.logger.log("operator", "resumed", {})

    def load_model(self, model_path):
        """Load trained PPO model."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 not installed")
        
        return PPO.load(model_path)

    def train_on_pair(self, prices: pd.DataFrame, x: str, y: str,
                      lookback: int = 252, timesteps: int = None,
                      stop_loss_threshold: float = 0.15,
                      early_stopping_threshold: float = None):
        """
        Train PPO model with enhanced features.
        
        Args:
            prices: Price dataframe
            x, y: Stock pair symbols
            lookback: Historical window
            timesteps: Training steps
            stop_loss_threshold: Stop-loss trigger
            early_stopping_threshold: Early stop if performance threshold met
        """
        if not self.active:
            return None
        
        if not GYMNASIUM_AVAILABLE or not SB3_AVAILABLE:
            print("Warning: Cannot train - dependencies not installed")
            return None

        if timesteps is None:
            timesteps = CONFIG["rl_timesteps"]

        series_x = prices[x]
        series_y = prices[y]
        
        # Create environment with enhancements
        env = PairTradingEnv(
            series_x, series_y, lookback, 
            test_mode=False,
            stop_loss_threshold=stop_loss_threshold
        )
        
        # Train with callback
        callback = PerformanceCallback(check_freq=1000) if SB3_AVAILABLE else None
        
        model = PPO(
            CONFIG["rl_policy"], 
            env, 
            verbose=0, 
            device="cpu",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )
        
        model.learn(total_timesteps=timesteps, callback=callback)

        model_path = os.path.join(self.storage_dir, f"operator_model_{x}_{y}.zip")
        model.save(model_path)

        # Evaluate trained model
        obs, _ = env.reset()
        done = False
        daily_returns = []
        positions_history = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            daily_returns.append(reward)
            positions_history.append(info.get('position', 0))

        # Calculate comprehensive metrics
        rets = np.array(daily_returns)
        rf_daily = 0.02 / 252
        excess_rets = rets - rf_daily

        mean_excess = np.mean(excess_rets)
        std_excess = np.std(excess_rets, ddof=1)
        sharpe = (mean_excess / (std_excess + 1e-8)) * np.sqrt(252)
        
        downside_rets = excess_rets[excess_rets < 0]
        if len(downside_rets) > 0:
            downside_std = np.std(downside_rets, ddof=1)
            sortino = (mean_excess / (downside_std + 1e-8)) * np.sqrt(252)
        else:
            sortino = np.inf

        # Calmar ratio
        calmar = (env.cum_returns / 100) / (env.max_drawdown + 1e-8)
        
        # Win rate
        winning_trades = np.sum(rets > 0)
        win_rate = winning_trades / len(rets) if len(rets) > 0 else 0
        
        # Average position size
        avg_position = np.mean(np.abs(positions_history))
        
        trace = {
            "pair": (x, y),
            "total_return_pct": env.cum_returns,
            "final_value": env.portfolio_value,
            "max_drawdown": env.max_drawdown,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "win_rate": win_rate,
            "avg_position": avg_position,
            "n_trades": len(daily_returns),
            "model_path": model_path
        }

        self.training_history.append(trace)
        self.logger.log("operator", "pair_trained", trace)
        self.message_bus.publish({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": "operator",
            "event": "pair_trained",
            "details": trace
        })

        return trace


def train_operator_on_pairs(operator: OperatorAgent, prices: pd.DataFrame, 
                            pairs: list, max_workers: int = 2):
    """Train operator on multiple pairs in parallel."""
    if not PARALLEL_TRAINING_AVAILABLE:
        print("Warning: Parallel training not available - training sequentially")
        return [operator.train_on_pair(prices, x, y) for x, y in pairs if operator.train_on_pair(prices, x, y)]
    
    all_traces = []
    
    def train(pair):
        x, y = pair
        print(f"\n🔹 Training Operator on pair ({x}, {y})")
        return operator.train_on_pair(prices, x, y)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train, pair) for pair in pairs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Operator Training"):
            result = f.result()
            if result:
                all_traces.append(result)

    save_path = os.path.join(operator.storage_dir, "all_operator_traces.json")
    with open(save_path, "w") as f:
        json.dump(all_traces, f, indent=2, default=str)
    
    operator.logger.log("operator", "batch_training_complete", {"n_pairs": len(all_traces)})
    print("\n✅ All traces saved successfully.")
    return all_traces
