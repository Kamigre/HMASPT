"""
Operator Agent for executing pairs trading strategies using Reinforcement Learning.
Requires: gymnasium, stable-baselines3 (optional)
"""

import os
import json
import time
import datetime
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import CONFIG
from utils import half_life, compute_spread
from agents.message_bus import MessageBus, JSONLogger


class PairTradingEnv(gym.Env):
    """
    Fixed PairTradingEnv with numerical stability improvements
    """

    def __init__(self, series_x: pd.Series, series_y: pd.Series, lookback: int = None,
                shock_prob: float = None, shock_scale: float = None,
                initial_capital: float = None, test_mode: bool = False):
    
        super().__init__()

        # Use CONFIG defaults if not provided
        if lookback is None:
            lookback = CONFIG.get("rl_lookback", 30)
        if shock_prob is None:
            shock_prob = CONFIG.get("shock_prob", 0.01)
        if shock_scale is None:
            shock_scale = CONFIG.get("shock_scale", 0.4)
        if initial_capital is None:
            initial_capital = CONFIG.get("initial_capital", 10000)

        self.align = pd.concat([series_x, series_y], axis=1).dropna()
        self.lookback = lookback
        self.test_mode = test_mode
        self.ptr = 0 if test_mode else lookback
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.position = 0
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.cum_returns = 0.0
        self.peak = initial_capital
        self.max_drawdown = 0.0
        self.trades = []
        self.shock_prob = shock_prob if not test_mode else 0.0
        self.shock_scale = shock_scale if not test_mode else 0.0
        
        # Compute spread
        self.spread = compute_spread(self.align.iloc[:, 0], self.align.iloc[:, 1])
        n = len(self.spread)
        shock_mask = np.random.rand(n) < self.shock_prob
        self.shocks = np.random.randn(n) * self.shock_scale * self.spread.std() * shock_mask
        self.spread_shocked = self.spread + self.shocks

        # Pre-compute features with stability fixes
        spread_mean = self.spread_shocked.rolling(self.lookback).mean()
        spread_std = self.spread_shocked.rolling(self.lookback).std()

        spread_std = spread_std
        self.zscores = (self.spread_shocked - spread_mean) / spread_std
        
        self.vols = self.spread_shocked.rolling(5).std()
        self.rx = self.align.iloc[:, 0].pct_change()
        self.ry = self.align.iloc[:, 1].pct_change()
        self.corrs = self.rx.rolling(5).corr(self.ry)
        
        # Convert to numpy and handle NaNs
        self.zscores_np = np.nan_to_num(self.zscores.to_numpy(), nan=0.0, posinf=5.0, neginf=-5.0)
        self.vols_np = np.nan_to_num(self.vols.to_numpy(), nan=1.0, posinf=1.0, neginf=0.0)
        self.corrs_np = np.nan_to_num(self.corrs.to_numpy(), nan=0.0, posinf=1.0, neginf=-1.0)
        self.spread_np = self.spread_shocked.to_numpy()
        
        # Normalize volatility for stable features
        self.vol_mean = np.nanmean(self.vols_np)
        self.vol_std = np.nanstd(self.vols_np)

    def _compute_features(self, idx: int):
        
        z = self.zscores_np[idx]
        vol = self.vols_np[idx]
        corr = self.corrs_np[idx]
        
        # Normalize volatility feature
        vol_normalized = (vol - self.vol_mean) / self.vol_std
        vol_normalized = np.clip(vol_normalized, -5, 5)
        
        # Use lookback window for half-life calculation
        start = max(0, idx - self.lookback)
        if idx > start:
            hl = half_life(self.spread_np[start:idx])
            if np.isnan(hl) or np.isinf(hl):
                hl = CONFIG.get("half_life_max", 100)
        else:
            hl = CONFIG.get("half_life_max", 100)
        
        # Normalize half-life to [0, 1] range
        hl_normalized = np.clip(hl / 100.0, 0, 1)
        
        features = np.array([z, vol_normalized, hl_normalized, corr], dtype=np.float32)
        
        # Final safety check for NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        self.ptr = 0 if self.test_mode else self.lookback
        self.position = 0
        self.portfolio_value = self.initial_capital
        self.cum_returns = 0.0
        self.peak = self.initial_capital
        self.max_drawdown = 0.0
        self.trades = []

        obs = self._compute_features(self.ptr)
        return obs, {}

    def step(self, action: int):
    
        target_pos = action - 1 
        next_ptr = self.ptr + 1
        terminated = next_ptr >= len(self.spread_np)
        truncated = False
    
        obs_next = self._compute_features(self.ptr)
    
        if terminated:
            return obs_next, 0.0, terminated, truncated, {
                "cum_reward": self.cum_returns,
                "max_drawdown": self.max_drawdown,
                "position": self.position,
                "pnl": 0.0,
                "return": 0.0
            }
    
        spread_now = self.spread_np[self.ptr]
        spread_next = self.spread_np[next_ptr]
        spread_change = spread_next - spread_now

        pnl = -self.position * spread_change

        # Compute how much you trade
        trade_size = abs(target_pos - self.position)

        # Apply 5 bps (0.05%) transaction cost proportional to trade size
        if trade_size > 0:
            cost_rate = CONFIG.get("transaction_cost", 0.0005)  # 5 bps
            pnl -= trade_size * cost_rate

        old_value = self.portfolio_value
        self.portfolio_value += pnl
    
        # Clip reward to prevent explosive values
        daily_return = pnl / max(1e-8, old_value)
        daily_return = np.clip(daily_return, -1, 1)  # Cap at ±100% daily return
    
        self.cum_returns = (self.portfolio_value / self.initial_capital - 1) * 100
        self.peak = max(self.peak, self.portfolio_value)
        self.max_drawdown = max(
            self.max_drawdown,
            (self.peak - self.portfolio_value) / max(self.peak, 1e-8)
        )
    
        self.position = target_pos
        self.ptr = next_ptr
        self.trades.append(daily_return)
    
        return obs_next, float(daily_return), terminated, truncated, {
            "pnl": float(pnl),
            "return": float(daily_return),
            "cum_reward": self.cum_returns,
            "max_drawdown": self.max_drawdown,
            "position": self.position
        }


@dataclass
class OperatorAgent:

    message_bus: MessageBus = None
    logger: JSONLogger = None
    storage_dir: str = "models/"

    def __post_init__(self):

        os.makedirs(self.storage_dir, exist_ok=True)
        self.active = True
        self.transaction_cost = CONFIG["transaction_cost"]

        self.current_step = 0  # Track current step in holdout execution
        self.traces_buffer = []  # Store all traces for supervisor access
        self.max_buffer_size = 1000  # Limit buffer size to prevent memory issues

    def get_current_step(self):

        return self.current_step

    def get_traces_since_step(self, start_step):

        return [t for t in self.traces_buffer if t.get('step', 0) >= start_step]

    def add_trace(self, trace):

        self.traces_buffer.append(trace)

        # Remove oldest traces if buffer exceeds limit
        if len(self.traces_buffer) > self.max_buffer_size:
            self.traces_buffer = self.traces_buffer[-self.max_buffer_size:]

    def clear_traces_before_step(self, step):

        self.traces_buffer = [t for t in self.traces_buffer if t.get('step', 0) >= step]

    def apply_command(self, command):

        cmd_type = command.get("command")

        if cmd_type == "pause":
            self.active = False
            self.logger.log("operator", "paused", {})

        elif cmd_type == "resume":
            self.active = True
            self.logger.log("operator", "resumed", {})

    def load_model(self, model_path):
        return PPO.load(model_path)

    # Individual pair training
    def train_on_pair(self, prices: pd.DataFrame, x: str, y: str,
                      lookback: int = None, timesteps: int = None,
                      shock_prob: float = None, shock_scale: float = None):

        if not self.active:
            return None

        # Use CONFIG defaults if not provided
        if lookback is None:
            lookback = CONFIG.get("rl_lookback", 30)
        if timesteps is None:
            timesteps = CONFIG.get("rl_timesteps", 20000)
        if shock_prob is None:
            shock_prob = CONFIG.get("shock_prob", 0.01)
        if shock_scale is None:
            shock_scale = CONFIG.get("shock_scale", 0.4)

        series_x = prices[x]
        series_y = prices[y]
        env = PairTradingEnv(series_x, series_y, lookback, shock_prob, shock_scale)
        model = PPO(CONFIG["rl_policy"], env, verbose=0, device="cpu")
        model.learn(total_timesteps=timesteps)

        model_path = os.path.join(self.storage_dir, f"operator_model_{x}_{y}.zip")
        model.save(model_path)

        obs, _ = env.reset()
        done = False
        daily_returns = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            daily_returns.append(reward)

        rets = np.array(daily_returns)
        rf_daily = CONFIG.get("risk_free_rate", 0.04) / 252
        excess_rets = rets - rf_daily

        sharpe = np.mean(excess_rets) / (np.std(excess_rets, ddof=1)) * np.sqrt(252)
        downside = excess_rets[excess_rets < 0]
        sortino = (np.mean(excess_rets) / (np.std(downside, ddof=1)) * np.sqrt(252)) if len(downside) else np.inf

        trace = {
            "pair": (x, y),
            "cum_reward": (np.prod(1 + rets) - 1) * 100,
            "max_drawdown": env.max_drawdown,
            "sharpe": sharpe,
            "sortino": sortino,
            "model_path": model_path
        }

        self.logger.log("operator", "pair_trained", trace)
        self.message_bus.publish({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": "operator",
            "event": "pair_trained",
            "details": trace
        })

        return trace


# Parallel pair training
def train_operator_on_pairs(operator: OperatorAgent, prices: pd.DataFrame, 
                        pairs: list, max_workers: int = None):

    if max_workers is None:
        max_workers = CONFIG.get("max_workers", 2)

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


def run_operator_holdout(operator, holdout_prices, pairs, supervisor, check_every_n_days=15):

    # Initialize operator state
    operator.traces_buffer = []
    operator.current_step = 0
    operator.evaluation_in_progress = False

    global_step = 0

    for pair in pairs:
        print(f"\n{'='*70}")
        print(f"Running pair: {pair[0]} - {pair[1]}")
        print(f"{'='*70}")

        # Validate pair exists in holdout data
        if pair[0] not in holdout_prices.columns or pair[1] not in holdout_prices.columns:
            print(f"⚠️ Warning: Tickers {pair} not found in holdout data - skipping")
            continue

        # Prepare aligned time series
        series_x = holdout_prices[pair[0]].dropna()
        series_y = holdout_prices[pair[1]].dropna()
        aligned = pd.concat([series_x, series_y], axis=1).dropna()

        print(f"  Data: {aligned.shape[0]} days")

        if len(aligned) < 2:
            print(f"⚠️ Insufficient data ({len(aligned)} days) - skipping pair")
            continue

        # Load trained model for this pair
        model_path = os.path.join(operator.storage_dir, f"operator_model_{pair[0]}_{pair[1]}.zip")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path} - skipping pair")
            continue

        model = operator.load_model(model_path)
        print(f"  ✓ Model loaded from {model_path}")

        # Create test environment (no shocks, test_mode=True)
        env = PairTradingEnv(
            series_x=aligned.iloc[:, 0],
            series_y=aligned.iloc[:, 1],
            lookback=CONFIG.get("rl_lookback", 30),
            shock_prob=0.0,  # No random shocks in evaluation
            shock_scale=0.0,
            initial_capital=CONFIG.get("initial_capital", 10000),
            test_mode=True  # Starts from beginning of data
        )

        # Execute trading episode
        episode_traces = []
        local_step = 0
        obs, info = env.reset()
        terminated = False
        truncated = False

        print(f"  Trading through {len(aligned)} days...")

        while not (terminated or truncated):
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action in environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Record trace for this step
            trace = {
                "pair": f"{pair[0]}-{pair[1]}",
                "step": global_step,
                "local_step": local_step,
                "reward": float(reward),
                "pnl": float(info.get("pnl", 0.0)),
                "return": float(info.get("return", 0.0)),
                "cum_reward": float(info.get("cum_reward", 0)),
                "position": float(info.get("position", 0)),
                "max_drawdown": float(info.get("max_drawdown", 0))
            }

            # Store trace in multiple places
            episode_traces.append(trace)
            operator.add_trace(trace)  # Uses the add_trace method for buffer management
            operator.logger.log("operator", "holdout_step", trace)

            local_step += 1
            global_step += 1
            operator.current_step = global_step

            # Optional: Add supervisor check at intervals
            if check_every_n_days > 0 and local_step % check_every_n_days == 0:
                # Supervisor could analyze recent traces and send commands
                # Example: supervisor.evaluate_operator_performance(operator)
                pass

            # Small delay for visualization/logging (remove for production)
            time.sleep(0.05)

        # Episode summary
        print(f"  ✓ Complete: {len(episode_traces)} steps")
        print(f"  Final cumulative return: {episode_traces[-1]['cum_reward']:.4f}")
        print(f"  Final P&L: {sum(t['pnl'] for t in episode_traces):.4f}")
        print(f"  Max drawdown: {max(t['max_drawdown'] for t in episode_traces):.4f}")

        # Calculate and log performance metrics
        sharpe = calculate_sharpe(episode_traces)
        sortino = calculate_sortino(episode_traces)

        # Log episode summary
        operator.logger.log("operator", "episode_complete", {
            "pair": f"{pair[0]}-{pair[1]}",
            "total_steps": len(episode_traces),
            "final_cum_return": episode_traces[-1]['cum_reward'],
            "total_pnl": sum(t['pnl'] for t in episode_traces),
            "max_drawdown": max(t['max_drawdown'] for t in episode_traces),
            "sharpe": sharpe,
            "sortino": sortino
        })

    print("\n" + "="*70)
    print("Holdout trading finished.")
    print(f"Total steps executed: {global_step}")
    print(f"Total traces collected: {len(operator.traces_buffer)}")
    print("="*70)
    
    return operator.traces_buffer


def calculate_sharpe(traces, risk_free_rate=None):
    """Calculate Sharpe ratio from episode traces."""
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    
    returns = [t['return'] for t in traces]
    if len(returns) == 0:
        return 0.0
    
    rf_daily = risk_free_rate / 252
    excess_returns = [r - rf_daily for r in returns]
    
    mean_excess = sum(excess_returns) / len(excess_returns)
    std_excess = (sum((r - mean_excess)**2 for r in excess_returns) / (len(excess_returns) - 1))**0.5
    
    if std_excess == 0:
        return 0.0
    
    return (mean_excess / std_excess) * (252**0.5)


def calculate_sortino(traces, risk_free_rate=None):
    """Calculate Sortino ratio from episode traces."""
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    
    returns = [t['return'] for t in traces]
    if len(returns) == 0:
        return 0.0
    
    rf_daily = risk_free_rate / 252
    excess_returns = [r - rf_daily for r in returns]
    downside_returns = [r for r in excess_returns if r < 0]
    
    if len(downside_returns) == 0:
        return float('inf')
    
    mean_excess = sum(excess_returns) / len(excess_returns)
    downside_std = (sum(r**2 for r in downside_returns) / (len(downside_returns) - 1))**0.5
    
    if downside_std == 0:
        return float('inf')
    
    return (mean_excess / downside_std) * (252**0.5)
