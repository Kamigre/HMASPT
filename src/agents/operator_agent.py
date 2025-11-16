"""
Operator Agent for executing pairs trading strategies using Reinforcement Learning.
Requires: gymnasium, stable-baselines3 (optional)
"""

import os
import json
import datetime
from dataclasses import dataclass
from typing import Optional
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
        Gymnasium environment for pairs trading with mean reversion.
        
        Features:
        - Observation: z-score, volatility, half-life, correlation
        - Actions: -1 (short), 0 (neutral), +1 (long)
        - Rewards: Based on spread changes minus transaction costs
        """
        
        metadata = {"render.modes": ["human", "plot"]}

        def __init__(self, series_x: pd.Series, series_y: pd.Series, lookback: int = 500,
                     shock_prob: float = 0.01, shock_scale: float = 0.1,
                     initial_capital: float = 1000):
            super().__init__()

            self.align = pd.concat([series_x, series_y], axis=1).dropna()
            self.lookback = lookback
            self.ptr = lookback

            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            self.action_space = spaces.Discrete(3)

            self.position = 0
            self.initial_capital = initial_capital
            self.portfolio_value = initial_capital
            self.cum_returns = 0.0
            self.peak = initial_capital
            self.max_drawdown = 0.0
            self.trades = []

            self.shock_prob = shock_prob
            self.shock_scale = shock_scale

            self.spread = compute_spread(self.align.iloc[:, 0], self.align.iloc[:, 1])
            n = len(self.spread)
            shock_mask = np.random.rand(n) < self.shock_prob
            self.shocks = np.random.randn(n) * self.shock_scale * self.spread.std() * shock_mask
            self.spread_shocked = self.spread + self.shocks

            self.zscores = (self.spread_shocked - self.spread_shocked.rolling(self.lookback).mean()) / self.spread_shocked.rolling(self.lookback).std()
            self.vols = self.spread_shocked.rolling(21).std()
            self.rx = self.align.iloc[:, 0].pct_change()
            self.ry = self.align.iloc[:, 1].pct_change()
            self.corrs = self.rx.rolling(21).corr(self.ry)

            self.zscores_np = np.nan_to_num(self.zscores.to_numpy())
            self.vols_np = np.nan_to_num(self.vols.to_numpy())
            self.corrs_np = np.nan_to_num(self.corrs.to_numpy())
            self.spread_np = self.spread_shocked.to_numpy()

        def _compute_features(self, idx: int):
            z = self.zscores_np[idx]
            vol = self.vols_np[idx]
            corr = self.corrs_np[idx]
            start = max(0, idx - self.lookback)
            hl = half_life(self.spread_np[start:idx]) if idx > start else CONFIG["half_life_max"]
            return np.array([z, vol, hl, corr], dtype=np.float32)

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)
            self.ptr = self.lookback
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
                    "cum_returns": self.cum_returns,
                    "max_drawdown": self.max_drawdown
                }

            ret = self.spread_np[next_ptr] - self.spread_np[self.ptr]
            reward = -ret * target_pos / (self.align.iloc[self.lookback, 0])

            if target_pos != self.position:
                reward -= CONFIG["transaction_cost"] + 0.002

            daily_return = reward / max(1e-8, self.portfolio_value)
            self.portfolio_value *= (1 + daily_return)
            self.cum_returns = self.portfolio_value - self.initial_capital
            self.peak = max(self.peak, self.portfolio_value)
            self.max_drawdown = max(self.max_drawdown, (self.peak - self.portfolio_value) / self.peak)

            self.position = target_pos
            self.ptr = next_ptr
            self.trades.append(daily_return)

            return obs_next, float(daily_return), terminated, truncated, {
                "pnl": daily_return,
                "cum_returns": self.cum_returns,
                "max_drawdown": self.max_drawdown
            }


@dataclass
class OperatorAgent:
    """
    Agent responsible for executing trades using RL-optimized strategies.
    
    Features:
    - Trains PPO models on pairs
    - Evaluates trading performance
    - Responds to supervisor commands
    """
    
    message_bus: MessageBus = None
    logger: JSONLogger = None
    storage_dir: str = "models/"

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        self.active = True
        self.transaction_cost = CONFIG["transaction_cost"]

    def apply_command(self, command):
        """Apply runtime commands from supervisor."""
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

    def train_on_pair(self, prices: pd.DataFrame, x: str, y: str,
                      lookback: int = 252, timesteps: int = None):
        """
        Train a PPO model on a stock pair.
        Returns performance metrics.
        """
        if not self.active:
            return None
        
        if not GYMNASIUM_AVAILABLE or not SB3_AVAILABLE:
            print("Warning: Cannot train - gymnasium or stable-baselines3 not installed")
            return None

        if timesteps is None:
            timesteps = CONFIG["rl_timesteps"]

        series_x = prices[x]
        series_y = prices[y]
        env = PairTradingEnv(series_x, series_y, lookback)
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
        rf_daily = 0.02 / 252
        excess_rets = rets - rf_daily

        sharpe = np.mean(excess_rets) / (np.std(excess_rets, ddof=1) + 1e-8) * np.sqrt(252)
        downside = excess_rets[excess_rets < 0]
        sortino = (np.mean(excess_rets) / (np.std(downside, ddof=1) + 1e-8) * np.sqrt(252)) if len(downside) else np.inf

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


def train_operator_on_pairs(operator: OperatorAgent, prices: pd.DataFrame, 
                            pairs: list, max_workers: int = 2):
    """
    Train operator on multiple pairs in parallel.
    Returns list of performance traces.
    """
    if not PARALLEL_TRAINING_AVAILABLE:
        print("Warning: Parallel training not available - will train sequentially")
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
