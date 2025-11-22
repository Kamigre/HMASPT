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

  def __init__(self, series_x: pd.Series, series_y: pd.Series, lookback: int = 30,
              shock_prob: float = 0.01, shock_scale: float = 0.2,
              initial_capital: float = 10000, test_mode: bool = False):
  
      super().__init__()

      self.align = pd.concat([series_x, series_y], axis=1).dropna()
      self.lookback = lookback
      self.test_mode = test_mode
      self.ptr = 0 if test_mode else lookback # In test mode, start from beginning; in training mode, start after lookback
      self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
      self.action_space = spaces.Discrete(3) # Buy, neutral or sell
      self.position = 0
      self.initial_capital = initial_capital
      self.portfolio_value = initial_capital
      self.cum_returns = 0.0
      self.peak = initial_capital
      self.max_drawdown = 0.0
      self.trades = []
      self.shock_prob = shock_prob if not test_mode else 0.0  # No shocks in test mode
      self.shock_scale = shock_scale if not test_mode else 0.0
      self.spread = compute_spread(self.align.iloc[:, 0], self.align.iloc[:, 1])
      n = len(self.spread)
      shock_mask = np.random.rand(n) < self.shock_prob
      self.shocks = np.random.randn(n) * self.shock_scale * self.spread.std() * shock_mask
      self.spread_shocked = self.spread + self.shocks

      # Pre-compute features
      self.zscores = (self.spread_shocked - self.spread_shocked.rolling(self.lookback).mean()) / \
                    self.spread_shocked.rolling(self.lookback).std()
      self.vols = self.spread_shocked.rolling(15).std()
      self.rx = self.align.iloc[:, 0].pct_change()
      self.ry = self.align.iloc[:, 1].pct_change()
      self.corrs = self.rx.rolling(15).corr(self.ry)
      self.zscores_np = np.nan_to_num(self.zscores.to_numpy())
      self.vols_np = np.nan_to_num(self.vols.to_numpy())
      self.corrs_np = np.nan_to_num(self.corrs.to_numpy())
      self.spread_np = self.spread_shocked.to_numpy()

  def _compute_features(self, idx: int):
      
      z = self.zscores_np[idx]
      vol = self.vols_np[idx]
      corr = self.corrs_np[idx]
      
      # Use lookback window for half-life calculation
      start = max(0, idx - self.lookback)
      hl = half_life(self.spread_np[start:idx]) if idx > start else CONFIG["half_life_max"]
      
      return np.array([z, vol, hl, corr], dtype=np.float32)

  def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

      super().reset(seed=seed)
      
      # In test mode, start from beginning; in training mode, start after lookback
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
  
      # Convert action (0,1,2) into target position (-1,0,+1)
      target_pos = action - 1 
      next_ptr = self.ptr + 1
      terminated = next_ptr >= len(self.spread_np)
      truncated = False
  
      # Next observation
      obs_next = self._compute_features(self.ptr)
  
      # If episode ends here, return terminal obs
      if terminated:
          return obs_next, 0.0, terminated, truncated, {
              "cum_reward": self.cum_returns,
              "max_drawdown": self.max_drawdown,
              "position": self.position,
              "pnl": 0.0,
              "return": 0.0
          }
  
      # Compute spread change
      spread_now = self.spread_np[self.ptr]
      spread_next = self.spread_np[next_ptr]
      spread_change = spread_next - spread_now
  
      # Compute PnL from position
      pnl = -self.position * spread_change
  
      # Apply transaction costs
      if target_pos != self.position:
          pnl -= CONFIG["transaction_cost"]
  
      # Update portfolio value
      old_value = self.portfolio_value
      self.portfolio_value += pnl
  
      # Compute true financial return
      daily_return = pnl / max(1e-8, old_value)
  
      # Update risk stats
      self.cum_returns = self.portfolio_value - self.initial_capital
      self.peak = max(self.peak, self.portfolio_value)
      self.max_drawdown = max(
          self.max_drawdown,
          (self.peak - self.portfolio_value) / self.peak
      )
  
      # Update position & pointer
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
                      lookback: int = 30, timesteps: int = 10000,
                      shock_prob: float = 0.01, shock_scale: float = 0.02):

        if not self.active:
            return None

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

# Parallel pair training
def train_operator_on_pairs(operator: OperatorAgent, prices: pd.DataFrame, 
                        pairs: list, max_workers: int = 2):

    all_traces = []

    def train(pair):
        x, y = pair
        print(f"\n🔹 Training Operator on pair ({x}, {y})")
        return operator.train_on_pair(prices, x, y, 30, 10000)

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
