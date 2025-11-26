"""
Operator Agent for executing pairs trading strategies using Reinforcement Learning.
FIXED VERSION with:
- Correct action mapping
- 10x position scaling for better learning signal
- 500k timesteps (increased from 20k)
- Better PPO hyperparameters
- Curriculum learning option
- No transaction costs during training (added back in testing)
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
from statsmodels.tsa.stattools import coint


class PairTradingEnv(gym.Env):

    def __init__(self, series_x: pd.Series, series_y: pd.Series, lookback: int = None,
                 shock_prob: float = None, shock_scale: float = None,
                 initial_capital: float = None, test_mode: bool = False,
                 position_scale: int = 10, enable_transaction_costs: bool = True):

        super().__init__()

        # Use CONFIG defaults if not provided
        if lookback is None:
            lookback = CONFIG.get("rl_lookback", 30)
        if shock_prob is None:
            shock_prob = CONFIG.get("shock_prob", 0.0)  # Default OFF
        if shock_scale is None:
            shock_scale = CONFIG.get("shock_scale", 0.0)
        if initial_capital is None:
            initial_capital = CONFIG.get("initial_capital", 10000)

        # Align series
        self.data = pd.concat([series_x, series_y], axis=1).dropna()
        self.lookback_short = lookback
        self.lookback_long = max(60, lookback * 2)
        self.test_mode = test_mode
        self.initial_capital = initial_capital
        self.position_scale = position_scale  # NEW: Scale positions
        self.enable_transaction_costs = enable_transaction_costs  # NEW: Toggle costs

        # Action: 3 discrete actions
        self.action_space = spaces.Discrete(3)

        # Observation: 15 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        # set shock behavior
        self.shock_prob = 0.0 if test_mode else shock_prob
        self.shock_scale = 0.0 if test_mode else shock_scale

        # Precompute features
        self._precompute_features()

        # Episode bookkeeping
        self.reset()

    def _precompute_features(self):
        x = self.data.iloc[:, 0]
        y = self.data.iloc[:, 1]

        # Spread
        self.spread = x - y

        # Add optional random shocks to spread for training
        n = len(self.spread)
        shock_mask = (np.random.rand(n) < self.shock_prob).astype(float)
        shocks = np.random.randn(n) * self.shock_scale * (self.spread.std() if self.spread.std() > 0 else 1.0) * shock_mask
        self.spread = self.spread + shocks

        # Timescale z-scores
        self.zscore_short = ((self.spread - self.spread.rolling(self.lookback_short).mean()) /
                             (self.spread.rolling(self.lookback_short).std() + 1e-8))
        self.zscore_long = ((self.spread - self.spread.rolling(self.lookback_long).mean()) /
                            (self.spread.rolling(self.lookback_long).std() + 1e-8))

        # Volatility
        self.vol_short = self.spread.rolling(self.lookback_short).std()
        self.vol_long = self.spread.rolling(self.lookback_long).std()
        self.vol_ratio = self.vol_short / (self.vol_long + 1e-8)

        # Momentum
        self.momentum_5d = self.spread.pct_change(5)
        self.momentum_15d = self.spread.pct_change(15)

        # Correlations
        rx = x.pct_change()
        ry = y.pct_change()
        self.corr_short = rx.rolling(self.lookback_short).corr(ry)
        self.corr_long = rx.rolling(self.lookback_long).corr(ry)

        # Autocorr (half-life proxy)
        self.autocorr = self.spread.rolling(self.lookback_short).apply(
            lambda s: s.autocorr() if len(s) > 1 else 0
        )

        # Spread percentile in recent history
        self.spread_percentile = self.spread.rolling(30).apply(
            lambda s: pd.Series(s).rank(pct=True).iloc[-1] if len(s) > 0 else 0.5
        )

        # Rolling cointegration p-values
        self.coint_pvalue = self._rolling_cointegration(x, y, window=90)

        # Normalize features
        self._normalize_features()

        # Convert to numpy
        self.spread_np = np.nan_to_num(self.spread.to_numpy(), nan=0.0, posinf=5.0, neginf=-5.0)
        self.vol_short_np = np.nan_to_num(self.vol_short.to_numpy(), nan=1.0, posinf=1.0, neginf=0.0)

    def _rolling_cointegration(self, x, y, window=90):
        pvalues = []
        for i in range(len(x)):
            if i < window:
                pvalues.append(0.5)
            else:
                try:
                    _, pval, _ = coint(x.iloc[i-window:i], y.iloc[i-window:i])
                    pvalues.append(pval)
                except Exception:
                    pvalues.append(0.5)
        return pd.Series(pvalues, index=x.index)

    def _normalize_features(self):
      features = [
          self.zscore_short, self.zscore_long,
          self.vol_ratio, self.momentum_5d, self.momentum_15d,
          self.corr_short, self.corr_long, self.autocorr,
          self.spread_percentile, self.coint_pvalue
      ]

      self.normalized_features = []

      for feat in features:
          fmin = feat.min()
          fmax = feat.max()
          rng = fmax - fmin

          if rng < 1e-8:
              normalized = feat - fmin
          else:
              normalized = (feat - fmin) / rng      # scale to [0, 1]
              normalized = 2 * normalized - 1       # scale to [-1, 1]

          self.normalized_features.append(normalized.fillna(0).to_numpy())

    def _get_observation(self, idx: int):
        if idx < 0 or idx >= len(self.spread_np):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs = np.array([
            float(self.normalized_features[0][idx]),  # zscore_short
            float(self.normalized_features[1][idx]),  # zscore_long
            float(self.normalized_features[2][idx]),  # vol_ratio
            float(self.normalized_features[3][idx]),  # momentum_5d
            float(self.normalized_features[4][idx]),  # momentum_15d
            float(self.normalized_features[5][idx]),  # corr_short
            float(self.normalized_features[6][idx]),  # corr_long
            float(self.normalized_features[7][idx]),  # autocorr
            float(self.normalized_features[8][idx]),  # spread_percentile
            float(self.normalized_features[9][idx]),  # coint_pvalue
            float(self.position / self.position_scale),  # normalized position
            float(self.portfolio_value / self.initial_capital - 1),  # return
            float(self.days_in_position),              # holding period
            float(self.spread_np[idx]),                # raw spread
            float(self.vol_short_np[idx])              # current volatility
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.idx = self.lookback_long if not self.test_mode else 0
        self.position = 0
        self.days_in_position = 0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.trades = []

        return self._get_observation(self.idx), {}

    def step(self, action: int):

        # Action 0 → Position -1 * scale (short)
        # Action 1 → Position  0 (flat)
        # Action 2 → Position +1 * scale (long)

        base_position = int(action) - 1
        target_position = base_position * self.position_scale  # Scale to -10, 0, +10

        current_idx = self.idx
        next_idx = current_idx + 1
        terminated = next_idx >= len(self.spread_np)

        if terminated:
            return self._get_observation(max(0, current_idx)), 0.0, True, False, {}

        current_spread = float(self.spread_np[current_idx])
        next_spread = float(self.spread_np[next_idx])
        spread_change = next_spread - current_spread

        # --------------------------
        # P&L calculation
        # --------------------------
        pnl = -self.position * spread_change

        # Transaction costs
        trade_size = abs(target_position - self.position)
        if trade_size > 0 and self.enable_transaction_costs:
            notional = trade_size * abs(current_spread)
            cost_rate = CONFIG.get("transaction_cost", 0.0005)
            transaction_cost = notional * cost_rate
            pnl -= transaction_cost
            self.days_in_position = 0
        else:
            self.days_in_position += 1

        previous_value = self.portfolio_value
        self.portfolio_value += pnl
        self.position = target_position
        self.idx = next_idx

        # Daily return
        daily_return = pnl / max(previous_value, 1e-8)

        # Track drawdown
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8)

        # --------------------------
        # NEW REWARD FUNCTION
        # --------------------------

        # 2. Mean-reversion signal from zscore_short
        # normalized_features[0] is the normalized z-score short
        signal = -self.normalized_features[0][self.idx]   # Correct mean-reversion direction
        mean_rev_bonus = 0.001 * signal * (target_position / self.position_scale)

        # Final reward
        reward = (
            daily_return
            + mean_rev_bonus
        )

        # --------------------------

        # Record trade
        self.trades.append({
            "pnl": float(pnl),
            "return": float(daily_return),
            "position": int(self.position)
        })

        obs = self._get_observation(self.idx)
        info = {
            "pnl": float(pnl),
            "return": float(daily_return),
            "position": int(self.position),
            "drawdown": float(drawdown),
            "cum_reward": float(self.portfolio_value / self.initial_capital - 1)
        }

        return obs, float(reward), terminated, False, info


@dataclass
class OperatorAgent:

    message_bus: MessageBus = None
    logger: JSONLogger = None
    storage_dir: str = "models/"

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        self.active = True
        self.transaction_cost = CONFIG.get("transaction_cost", 0.0005)
        self.current_step = 0
        self.traces_buffer = []
        self.max_buffer_size = 1000

    def get_current_step(self):
        return self.current_step

    def get_traces_since_step(self, start_step):
        return [t for t in self.traces_buffer if t.get('step', 0) >= start_step]

    def add_trace(self, trace):
        self.traces_buffer.append(trace)
        if len(self.traces_buffer) > self.max_buffer_size:
            self.traces_buffer = self.traces_buffer[-self.max_buffer_size:]

    def clear_traces_before_step(self, step):
        self.traces_buffer = [t for t in self.traces_buffer if t.get('step', 0) >= step]

    def apply_command(self, command):
        cmd_type = command.get("command")
        if cmd_type == "pause":
            self.active = False
            if self.logger:
                self.logger.log("operator", "paused", {})
        elif cmd_type == "resume":
            self.active = True
            if self.logger:
                self.logger.log("operator", "resumed", {})

    def load_model(self, model_path):
        return PPO.load(model_path)

    def train_on_pair(self, prices: pd.DataFrame, x: str, y: str,
                      lookback: int = None, timesteps: int = None,
                      shock_prob: float = None, shock_scale: float = None,
                      use_curriculum: bool = False):
        """
        IMPROVED training with:
        - 500k timesteps (was 20k)
        - Better PPO hyperparameters
        - Position scaling (10x)
        - Optional curriculum learning
        """

        if not self.active:
            return None

        # IMPROVED DEFAULTS
        if lookback is None:
            lookback = CONFIG.get("rl_lookback", 20)  # Shorter = more responsive
        if timesteps is None:
            timesteps = CONFIG.get("rl_timesteps", 500000)  # INCREASED 25x
        if shock_prob is None:
            shock_prob = 0.0  # OFF by default
        if shock_scale is None:
            shock_scale = 0.0

        series_x = prices[x]
        series_y = prices[y]

        print(f"\n{'='*70}")
        print(f"Training pair: {x} - {y}")
        print(f"  Data length: {len(series_x)} days")
        print(f"  Timesteps: {timesteps:,}")
        print(f"  Position scale: 10x")
        print(f"  Curriculum learning: {use_curriculum}")
        print(f"{'='*70}")

        if use_curriculum:
            # CURRICULUM LEARNING: Start easy, get harder
            print("\n🎓 Stage 1/3: Training without transaction costs...")
            env = PairTradingEnv(
                series_x, series_y, lookback, shock_prob, shock_scale,
                position_scale=10, enable_transaction_costs=False
            )
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,  # Encourage exploration
                verbose=0,
                device="cpu"
            )
            model.learn(total_timesteps=timesteps // 3)

            print("\n🎓 Stage 2/3: Adding 50% transaction costs...")
            original_cost = CONFIG.get("transaction_cost", 0.0005)
            CONFIG["transaction_cost"] = original_cost * 0.5
            env = PairTradingEnv(
                series_x, series_y, lookback, shock_prob, shock_scale,
                position_scale=10, enable_transaction_costs=True
            )
            model.set_env(env)
            model.learn(total_timesteps=timesteps // 3)

            print("\n🎓 Stage 3/3: Full transaction costs...")
            CONFIG["transaction_cost"] = original_cost
            env = PairTradingEnv(
                series_x, series_y, lookback, shock_prob, shock_scale,
                position_scale=10, enable_transaction_costs=True
            )
            model.set_env(env)
            model.learn(total_timesteps=timesteps // 3)

        else:
            # STANDARD TRAINING (no transaction costs)
            print("\n🚀 Training with standard approach (no costs)...")
            env = PairTradingEnv(
                series_x, series_y, lookback, shock_prob, shock_scale,
                position_scale=10, enable_transaction_costs=False  # Train without costs
            )
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01,
                verbose=1,  # Show progress
                device="cpu"
            )
            model.learn(total_timesteps=timesteps)

        # Save model
        model_path = os.path.join(self.storage_dir, f"operator_model_{x}_{y}.zip")
        model.save(model_path)
        print(f"\n✅ Model saved to {model_path}")

        # Evaluate on training data (without costs for fair comparison)
        print("\n📊 Evaluating on training data...")
        env_eval = PairTradingEnv(
            series_x, series_y, lookback, 0.0, 0.0,
            position_scale=10, enable_transaction_costs=False,
            test_mode=False
        )
        
        obs, _ = env_eval.reset()
        done = False
        daily_returns = []
        positions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env_eval.step(action)
            daily_returns.append(info.get('return', 0))
            positions.append(info.get('position', 0))

        # Calculate metrics
        rets = np.array(daily_returns)
        rf_daily = CONFIG.get("risk_free_rate", 0.04) / 252
        excess_rets = rets - rf_daily

        sharpe = 0.0
        if len(excess_rets) > 1 and np.std(excess_rets, ddof=1) > 1e-8:
            sharpe = np.mean(excess_rets) / np.std(excess_rets, ddof=1) * np.sqrt(252)
        
        downside = excess_rets[excess_rets < 0]
        sortino = 0.0
        if len(downside) > 1 and np.std(downside, ddof=1) > 1e-8:
            sortino = np.mean(excess_rets) / np.std(downside, ddof=1) * np.sqrt(252)

        final_return = (env_eval.portfolio_value / env_eval.initial_capital - 1) * 100

        # Position analysis
        unique_positions = np.unique(positions)
        print(f"\n📈 Training Results:")
        print(f"  Final Return: {final_return:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        print(f"  Sortino Ratio: {sortino:.3f}")
        print(f"  Positions used: {unique_positions}")
        for pos in unique_positions:
            count = np.sum(np.array(positions) == pos)
            pct = count / len(positions) * 100
            print(f"    Position {int(pos)}: {pct:.1f}% of time")

        trace = {
            "pair": (x, y),
            "cum_reward": final_return,
            "max_drawdown": (env_eval.peak_value - env_eval.portfolio_value) / env_eval.peak_value,
            "sharpe": sharpe,
            "sortino": sortino,
            "model_path": model_path,
            "positions_used": unique_positions.tolist()
        }

        if self.logger:
            self.logger.log("operator", "pair_trained", trace)
        if self.message_bus:
            self.message_bus.publish({
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "agent": "operator",
                "event": "pair_trained",
                "details": trace
            })

        return trace


def train_operator_on_pairs(operator: OperatorAgent, prices: pd.DataFrame,
                        pairs: list, max_workers: int = None, use_curriculum: bool = False):

    if max_workers is None:
        max_workers = CONFIG.get("max_workers", 2)

    all_traces = []

    def train(pair):
        x, y = pair
        return operator.train_on_pair(prices, x, y, use_curriculum=use_curriculum)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train, pair) for pair in pairs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Operator Training"):
            result = f.result()
            if result:
                all_traces.append(result)

    save_path = os.path.join(operator.storage_dir, "all_operator_traces.json")
    with open(save_path, "w") as f:
        json.dump(all_traces, f, indent=2, default=str)

    if operator.logger:
        operator.logger.log("operator", "batch_training_complete", {"n_pairs": len(all_traces)})
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for trace in all_traces:
        print(f"{trace['pair'][0]}-{trace['pair'][1]}: "
              f"Return={trace['cum_reward']:.2f}%, Sharpe={trace['sharpe']:.2f}")
    print("="*70)
    
    return all_traces


def run_operator_holdout(operator, holdout_prices, pairs, supervisor, check_every_n_days=15):
    """
    TESTING: Apply transaction costs here (realistic evaluation)
    """

    operator.traces_buffer = []
    operator.current_step = 0
    operator.evaluation_in_progress = False

    global_step = 0

    for pair in pairs:
        print(f"\n{'='*70}")
        print(f"Testing pair: {pair[0]} - {pair[1]}")
        print(f"{'='*70}")

        if pair[0] not in holdout_prices.columns or pair[1] not in holdout_prices.columns:
            print(f"⚠️ Warning: Tickers {pair} not found in holdout data - skipping")
            continue

        series_x = holdout_prices[pair[0]].dropna()
        series_y = holdout_prices[pair[1]].dropna()
        aligned = pd.concat([series_x, series_y], axis=1).dropna()

        print(f"  Data: {aligned.shape[0]} days")

        if len(aligned) < 2:
            print(f"⚠️ Insufficient data - skipping")
            continue

        model_path = os.path.join(operator.storage_dir, f"operator_model_{pair[0]}_{pair[1]}.zip")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found - skipping")
            continue

        model = operator.load_model(model_path)
        print(f"  ✓ Model loaded")

        # TEST ENVIRONMENT: With transaction costs
        env = PairTradingEnv(
            series_x=aligned.iloc[:, 0],
            series_y=aligned.iloc[:, 1],
            lookback=CONFIG.get("rl_lookback", 20),
            shock_prob=0.0,
            shock_scale=0.0,
            initial_capital=CONFIG.get("initial_capital", 10000),
            test_mode=True,
            position_scale=10,
            enable_transaction_costs=True  # Costs enabled in testing
        )

        episode_traces = []
        local_step = 0
        obs, info = env.reset()
        terminated = False

        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)

            trace = {
                "pair": f"{pair[0]}-{pair[1]}",
                "step": global_step,
                "local_step": local_step,
                "reward": float(reward),
                "pnl": float(info.get("pnl", 0.0)),
                "return": float(info.get("return", 0.0)),
                "cum_reward": float(info.get("cum_reward", 0)),
                "position": float(info.get("position", 0)),
                "max_drawdown": float(info.get("drawdown", 0))
            }

            episode_traces.append(trace)
            operator.add_trace(trace)
            if operator.logger:
                operator.logger.log("operator", "holdout_step", trace)

            local_step += 1
            global_step += 1
            operator.current_step = global_step

        print(f"  ✓ Complete: {len(episode_traces)} steps")
        if len(episode_traces) > 0:
            print(f"  Final return: {episode_traces[-1]['cum_reward']*100:.2f}%")
            print(f"  Total P&L: ${sum(t['pnl'] for t in episode_traces):.2f}")

        sharpe = calculate_sharpe(episode_traces)
        sortino = calculate_sortino(episode_traces)

        if operator.logger:
            operator.logger.log("operator", "episode_complete", {
                "pair": f"{pair[0]}-{pair[1]}",
                "total_steps": len(episode_traces),
                "final_cum_return": episode_traces[-1]['cum_reward'] if episode_traces else 0,
                "total_pnl": sum(t['pnl'] for t in episode_traces),
                "sharpe": sharpe,
                "sortino": sortino
            })

    print("\n" + "="*70)
    print("Holdout testing complete")
    print(f"Total steps: {global_step}")
    print("="*70)

    return operator.traces_buffer


def calculate_sharpe(traces, risk_free_rate=None):
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)

    returns = [t['return'] for t in traces]
    if len(returns) == 0:
        return 0.0

    rf_daily = risk_free_rate / 252
    excess_returns = [r - rf_daily for r in returns]

    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1) if len(excess_returns) > 1 else 0.0

    if std_excess == 0:
        return 0.0

    return (mean_excess / std_excess) * np.sqrt(252)


def calculate_sortino(traces, risk_free_rate=None):
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

    mean_excess = np.mean(excess_returns)
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0

    if downside_std == 0:
        return float('inf')

    return (mean_excess / downside_std) * np.sqrt(252)
