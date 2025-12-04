import os
import json
import time
import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any
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
from agents.message_bus import JSONLogger
from statsmodels.tsa.stattools import coint

class PairTradingEnv(gym.Env):

    def __init__(self, series_x: pd.Series, series_y: pd.Series, 
                 lookback: int = 30,
                 initial_capital: float = 10000,
                 position_scale: int = 100,
                 transaction_cost_rate: float = 0.0005,
                 test_mode: bool = False):
        
        super().__init__()
        
        # Align series
        self.data = pd.concat([series_x, series_y], axis=1).dropna()
        self.lookback = lookback
        self.test_mode = test_mode
        self.initial_capital = initial_capital
        self.position_scale = position_scale
        self.transaction_cost_rate = transaction_cost_rate
        
        # Action: 3 discrete actions (short, flat, long)
        self.action_space = spaces.Discrete(3)
        
        # Observation space (simplified for clarity)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        
        # Precompute spread and features
        self._precompute_features()
        
        self.reset()

    def _precompute_features(self):
        """Compute spread and basic features"""
        x = self.data.iloc[:, 0]
        y = self.data.iloc[:, 1]
        
        # Raw spread
        self.spread = x - y
        
        # Z-scores at different timescales
        self.zscore_short = (
            (self.spread - self.spread.rolling(self.lookback).mean()) /
            (self.spread.rolling(self.lookback).std() + 1e-8)
        )
        
        self.zscore_long = (
            (self.spread - self.spread.rolling(self.lookback * 2).mean()) /
            (self.spread.rolling(self.lookback * 2).std() + 1e-8)
        )
        
        # Volatility
        self.vol = self.spread.rolling(self.lookback).std()
        
        # Convert to numpy
        self.spread_np = np.nan_to_num(self.spread.to_numpy(), nan=0.0)
        self.zscore_short_np = np.nan_to_num(self.zscore_short.to_numpy(), nan=0.0)
        self.zscore_long_np = np.nan_to_num(self.zscore_long.to_numpy(), nan=0.0)
        self.vol_np = np.nan_to_num(self.vol.to_numpy(), nan=1.0)

    def _get_observation(self, idx: int) -> np.ndarray:
        """Build observation vector"""
        if idx < 0 or idx >= len(self.spread_np):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        obs = np.array([
            self.zscore_short_np[idx],
            self.zscore_long_np[idx],
            self.vol_np[idx],
            self.spread_np[idx],
            float(self.position / self.position_scale),  # normalized position
            float(self.entry_spread) if self.position != 0 else 0.0,
            float(self.unrealized_pnl),
            float(self.realized_pnl),
            float(self.cash / self.initial_capital - 1),  # cash return
            float(self.portfolio_value / self.initial_capital - 1),  # total return
            float(self.days_in_position),
            float(self.num_trades),
        ], dtype=np.float32)
        
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.idx = self.lookback if not self.test_mode else 0
        self.position = 0
        self.entry_spread = 0.0  # Spread level when position was entered
        self.days_in_position = 0
        
        # Financial tracking
        self.cash = self.initial_capital
        self.realized_pnl = 0.0  # P&L from closed trades
        self.unrealized_pnl = 0.0  # Mark-to-market of open position
        self.portfolio_value = self.initial_capital
        
        # Performance tracking
        self.peak_value = self.initial_capital
        self.num_trades = 0
        self.trade_history = []
        
        return self._get_observation(self.idx), {}

    def step(self, action: int):
        """
        Execute one trading step with proper P&L calculation.
        
        Action mapping:
        0 → Short position (-1 * scale)
        1 → Flat (0)
        2 → Long position (+1 * scale)
        """
        
        # Map action to target position
        base_position = int(action) - 1  # Maps to -1, 0, +1
        target_position = base_position * self.position_scale
        
        current_idx = self.idx
        
        # Check if episode is done
        terminated = current_idx >= len(self.spread_np) - 1
        if terminated:
            obs = self._get_observation(current_idx)
            info = {
                'portfolio_value': float(self.portfolio_value),
                'cash': float(self.cash),
                'realized_pnl': float(self.realized_pnl),
                'unrealized_pnl': float(self.unrealized_pnl),
                'realized_pnl_this_step': 0.0,
                'transaction_costs': 0.0,
                'position': int(self.position),
                'entry_spread': float(self.entry_spread),
                'current_spread': float(self.spread_np[current_idx]),
                'days_in_position': int(self.days_in_position),
                'daily_return': 0.0,
                'drawdown': (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8),
                'num_trades': int(self.num_trades),
                'trade_occurred': False,
                'cum_return': float(self.portfolio_value / self.initial_capital - 1)
            }
            return obs, 0.0, True, False, info

        next_idx = current_idx + 1
        
        # Get current and next spread
        current_spread = float(self.spread_np[current_idx])
        next_spread = float(self.spread_np[next_idx])
        
        # ============================================================
        # POSITION CHANGE LOGIC
        # ============================================================
        
        position_change = target_position - self.position
        trade_occurred = (position_change != 0)
        
        realized_pnl_this_step = 0.0
        transaction_costs = 0.0
        
        if trade_occurred:
            # --------------------------------------------------------
            # CASE 1: Closing or reducing a position → Realize P&L
            # --------------------------------------------------------
            if self.position != 0:
                # Calculate P&L on the portion being closed
                spread_change = current_spread - self.entry_spread
                
                if target_position == 0:
                    # Fully closing position
                    closed_size = abs(self.position)
                    pnl_on_closed = self.position * spread_change
                    
                elif np.sign(target_position) == np.sign(self.position):
                    # Reducing position (same direction)
                    closed_size = abs(position_change)
                    pnl_on_closed = position_change * spread_change
                    
                else:
                    # Flipping position (e.g., from long to short)
                    # Close entire old position, then open new one
                    closed_size = abs(self.position)
                    pnl_on_closed = self.position * spread_change
                
                realized_pnl_this_step += pnl_on_closed
                
                # Record trade
                self.trade_history.append({
                    'entry_spread': self.entry_spread,
                    'exit_spread': current_spread,
                    'position': self.position,
                    'pnl': pnl_on_closed,
                    'holding_days': self.days_in_position
                })
            
            # --------------------------------------------------------
            # CASE 2: Opening or increasing position → Set entry price
            # --------------------------------------------------------
            if target_position != 0:
                if np.sign(target_position) != np.sign(self.position):
                    # New position or flip → reset entry price
                    self.entry_spread = current_spread
                    self.days_in_position = 0
                # If increasing same direction, keep original entry price
            else:
                # Flat position
                self.entry_spread = 0.0
                self.days_in_position = 0
            
            # --------------------------------------------------------
            # Transaction costs
            # --------------------------------------------------------
            trade_size = abs(position_change)
            notional = trade_size * abs(current_spread)
            transaction_costs = notional * self.transaction_cost_rate
            
            self.num_trades += 1
            
        else:
            # No trade occurred
            self.days_in_position += 1
        
        # ============================================================
        # UPDATE FINANCIAL STATE
        # ============================================================
        
        # Update position
        self.position = target_position
        
        # Update realized P&L (subtract transaction costs)
        self.realized_pnl += realized_pnl_this_step - transaction_costs
        
        # Update cash
        self.cash = self.initial_capital + self.realized_pnl
        
        # Calculate unrealized P&L on current position
        if self.position != 0:
            self.unrealized_pnl = self.position * (next_spread - self.entry_spread)
        else:
            self.unrealized_pnl = 0.0
        
        # Total portfolio value
        self.portfolio_value = self.cash + self.unrealized_pnl
        
        # Update index
        self.idx = next_idx
        
        # ============================================================
        # PERFORMANCE METRICS
        # ============================================================
        
        # Daily return (on total portfolio value)
        previous_value = self.cash - realized_pnl_this_step + transaction_costs + \
                        (self.position * (current_spread - self.entry_spread))
        daily_return = (self.portfolio_value - previous_value) / max(previous_value, 1e-8)
        
        # Drawdown
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8)
        
        # ============================================================
        # REWARD FUNCTION
        # ============================================================
        
        # Main objective: portfolio return
        reward = 10.0 * daily_return
        
        # Penalize drawdown
        reward -= 5.0 * (drawdown ** 2)
        
        # Small penalty for holding time (encourage mean reversion trading)
        if self.position != 0:
            reward -= 0.00 * self.days_in_position
        
        # Bonus for realized profits
        if realized_pnl_this_step > 0:
            reward += 0.5 * (realized_pnl_this_step / self.initial_capital)
        
        reward -= 0.0005 * abs(position_change)
        
        # ============================================================
        # INFO DICT
        # ============================================================
        
        info = {
            'portfolio_value': float(self.portfolio_value),
            'cash': float(self.cash),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl_this_step': float(realized_pnl_this_step),
            'transaction_costs': float(transaction_costs),
            'position': int(self.position),
            'entry_spread': float(self.entry_spread),
            'current_spread': float(next_spread),
            'days_in_position': int(self.days_in_position),
            'daily_return': float(daily_return),
            'drawdown': float(drawdown),
            'num_trades': int(self.num_trades),
            'trade_occurred': bool(trade_occurred),
            'cum_return': float(self.portfolio_value / self.initial_capital - 1)
        }
        
        obs = self._get_observation(self.idx)
        
        return obs, float(reward), terminated, False, info

@dataclass
class OperatorAgent:
  
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

        if not self.active:
            return None

        # Defaults
        if lookback is None:
            lookback = CONFIG.get("rl_lookback", 20)
        if timesteps is None:
            timesteps = CONFIG.get("rl_timesteps", 500000)
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
        print(f"  Position scale: 100x")
        print(f"{'='*70}")

        print("\n🚀 Training with standard approach (no costs)...")
        env = PairTradingEnv(
              series_x, series_y, lookback, position_scale=100,
              transaction_cost_rate = 0.0005, test_mode=False
          )

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0001,
            n_steps=512,
            batch_size=64,
            n_epochs=20,
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

        # Evaluate on training data
        print("\n📊 Evaluating on training data...")
        env_eval = PairTradingEnv(
              series_x, series_y, lookback, position_scale=100,
              transaction_cost_rate = 0.0005, test_mode=False
        )
        
        obs, _ = env_eval.reset()
        done = False
        daily_returns = []
        positions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env_eval.step(action)
            daily_returns.append(info.get('daily_return', 0))
            positions.append(info.get('position', 0))

        # Calculate metrics
        rets = np.array(daily_returns)
        rf_daily = CONFIG.get("risk_free_rate", 0.04) / 252
        excess_rets = rets - rf_daily

        sharpe = 0.0
        if len(excess_rets) > 1 and np.std(excess_rets, ddof=1) > 1e-8:
            sharpe = np.mean(excess_rets) / np.std(excess_rets, ddof=1) * np.sqrt(252)
            print(np.mean(excess_rets))
            print(np.std(excess_rets, ddof=1))
        
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
            "cum_return": final_return,
            "max_drawdown": (env_eval.peak_value - env_eval.portfolio_value) / env_eval.peak_value,
            "sharpe": sharpe,
            "sortino": sortino,
            "model_path": model_path,
            "positions_used": unique_positions.tolist()
        }

        if self.logger:
            self.logger.log("operator", "pair_trained", trace)

        return trace


def train_operator_on_pairs(operator: OperatorAgent, prices: pd.DataFrame,
                        pairs: list, max_workers: int = None):

    if max_workers is None:
        max_workers = CONFIG.get("max_workers", 2)

    all_traces = []

    def train(pair):
        x, y = pair
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

    if operator.logger:
        operator.logger.log("operator", "batch_training_complete", {"n_pairs": len(all_traces)})
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for trace in all_traces:
        print(f"{trace['pair'][0]}-{trace['pair'][1]}: "
              f"Return={trace['cum_return']:.2f}%, Sharpe={trace['sharpe']:.2f}")
    print("="*70)
    
    return all_traces


def run_operator_holdout(operator, holdout_prices, pairs, supervisor):
    """
    Run holdout testing with supervisor monitoring.
    
    Args:
        operator: OperatorAgent instance
        holdout_prices: DataFrame with holdout price data
        pairs: List of (x, y) tuples to test
        supervisor: SupervisorAgent instance
    
    Returns:
        all_traces: List of all trading step traces
        skipped_pairs: List of pairs that were stopped early by supervisor
    """
    
    # Get check_interval from CONFIG
    if "supervisor_rules" in CONFIG and "holdout" in CONFIG["supervisor_rules"]:
        check_interval = CONFIG["supervisor_rules"]["holdout"].get("check_interval", 20)
    else:
        check_interval = 20  # Fallback default
    
    operator.traces_buffer = []
    operator.current_step = 0
    operator.evaluation_in_progress = False

    global_step = 0
    all_traces = []
    skipped_pairs = []

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
              series_x=aligned.iloc[:, 0], series_y=aligned.iloc[:, 1], 
              lookback=CONFIG.get("rl_lookback", 20), position_scale=100,
              transaction_cost_rate = 0.0005, test_mode=True
          )

        episode_traces = []
        local_step = 0
        obs, info = env.reset()
        terminated = False
        skip_to_next_pair = False

        # Trading loop with supervisor monitoring
        while not terminated and not skip_to_next_pair:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)

            trace = {
                "pair": f"{pair[0]}-{pair[1]}",
                "step": global_step,
                "local_step": local_step,
                "reward": float(reward),
                "portfolio_value": float(info.get("portfolio_value", 0.0)),
                "cum_return": float(info.get("cum_return", 0.0)),
                "cum_reward": float(info.get("cum_reward", 0.0)),
                "position": float(info.get("position", 0)),
                "max_drawdown": float(info.get("drawdown", 0)),
                "cash": float(info.get("cash", 0.0)),
                "realized_pnl": float(info.get("realized_pnl", 0.0)),
                "unrealized_pnl": float(info.get("unrealized_pnl", 0.0)),
                "realized_pnl_this_step": float(info.get("realized_pnl_this_step", 0.0)),
                "transaction_costs": float(info.get("transaction_costs", 0.0)),
                "entry_spread": float(info.get("entry_spread", 0.0)),
                "current_spread": float(info.get("current_spread", 0.0)),
                "days_in_position": int(info.get("days_in_position", 0)),
                "daily_return": float(info.get("daily_return", 0.0)),
                "num_trades": int(info.get("num_trades", 0)),
                "trade_occurred": bool(info.get("trade_occurred", False)),
            }

            episode_traces.append(trace)
            all_traces.append(trace)
            operator.add_trace(trace)
            
            # Save detailed trace for visualization
            if hasattr(operator, 'save_detailed_trace'):
                operator.save_detailed_trace(trace)
            
            if operator.logger:
                operator.logger.log("operator", "holdout_step", trace)

            # ============================================================
            # SUPERVISOR MONITORING (every N steps)
            # ============================================================
            if local_step > 0 and local_step % check_interval == 0:
                # Call supervisor with phase parameter
                decision = supervisor.check_operator_performance(
                    episode_traces, 
                    pair,
                    phase="holdout"
                )
                
                if decision["action"] == "stop":
                    severity = decision.get("severity", "critical")
                    print(f"\n⛔ SUPERVISOR INTERVENTION [{severity.upper()}]: Skipping to next pair")
                    print(f"   Reason: {decision['reason']}")
                    print(f"   Metrics: {decision['metrics']}")
                    print(f"   Steps completed: {local_step}/{len(aligned)}")
                    
                    # Record skip information
                    skip_info = {
                        "pair": f"{pair[0]}-{pair[1]}",
                        "reason": decision['reason'],
                        "severity": severity,
                        "step_stopped": global_step,
                        "local_step_stopped": local_step,
                        "metrics": decision['metrics']
                    }
                    
                    skipped_pairs.append(skip_info)
                    skip_to_next_pair = True
                    
                    # Log the intervention
                    if operator.logger:
                        operator.logger.log("supervisor", "intervention", skip_info)
                    
                    continue
                
                elif decision["action"] == "adjust":
                    print(f"\n⚠️  SUPERVISOR WARNING [{decision.get('severity', 'warning').upper()}]:")
                    print(f"   {decision['reason']}")
                    if 'suggestion' in decision:
                        print(f"   💡 Suggestion: {decision['suggestion']}")
                
                elif decision["action"] == "warn":
                    # Only show warnings occasionally (every 4 checks)
                    if local_step % (check_interval * 4) == 0:
                        print(f"\nℹ️  SUPERVISOR INFO:")
                        print(f"   {decision['reason']}")
                
                # Show periodic performance updates
                if local_step % (check_interval * 2) == 0:
                    metrics = decision["metrics"]
                    print(f"\n📊 Step {local_step}: DD={metrics.get('drawdown', 0):.2%}, "
                          f"Sharpe={metrics.get('sharpe', 0):.2f}, "
                          f"WinRate={metrics.get('win_rate', 0):.2%}")

            local_step += 1
            global_step += 1
            operator.current_step = global_step

        # ============================================================
        # END OF PAIR SUMMARY
        # ============================================================
        
        if skip_to_next_pair:
            print(f"⏭️  Pair skipped early at step {local_step}")
        else:
            print(f"  ✓ Complete: {len(episode_traces)} steps")
        
        # Extract values for metrics
        daily_returns = np.array([t["daily_return"] for t in episode_traces])
        pnls = np.array([t["realized_pnl_this_step"] for t in episode_traces])
        positions = np.array([t["position"] for t in episode_traces])

        # ---------- TRADE DETECTION ----------
        trades = []
        last_position = 0
        for t in episode_traces:
            pos = t["position"]
            pnl = t.get("realized_pnl_this_step", 0)
            if pos != last_position and pnl != 0:  # only count trades with non-zero PnL
                trades.append({"position": pos, "realized_pnl": pnl})
            last_position = pos

        pnls_list = [tr["realized_pnl"] for tr in trades]

        n_trades = len(pnls_list)
        wins = [1 for pnl in pnls_list if pnl > 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_trade_pnl = np.mean(pnls_list) if n_trades > 0 else 0.0

        # ---------- POSITION USAGE ----------
        unique_positions, pos_counts = np.unique(positions, return_counts=True)
        pos_usage = {
            int(p): float(c) / len(positions)
            for p, c in zip(unique_positions, pos_counts)
        }

        # ---------- RETURN METRICS ----------
        # Filter daily returns to remove zeros
        filtered_returns = [t.get("daily_return", 0) for t in episode_traces if t.get("daily_return", 0) != 0]

        ret_mean = np.mean(filtered_returns) if filtered_returns else 0.0
        ret_std = np.std(filtered_returns) if filtered_returns else 0.0
        ret_median = np.median(filtered_returns) if filtered_returns else 0.0

        # ---------- DRAWDOWN ----------
        max_dd = max(t.get("max_drawdown", 0) for t in episode_traces) if episode_traces else 0

        # Sharpe and Sortino
        sharpe = calculate_sharpe(episode_traces)
        sortino = calculate_sortino(episode_traces)

        # ============================
        # STORE IN LOGGING / TRACE
        # ============================
        extra_stats = {
            "trades": n_trades,
            "win_rate": win_rate,
            "avg_trade_pnl": avg_trade_pnl,
            "position_usage": pos_usage,
            "return_mean": ret_mean,
            "return_std": ret_std,
            "return_median": ret_median,
            "max_drawdown": max_dd,
            "was_skipped": skip_to_next_pair
        }

        print(f"  Trades: {n_trades}")
        print(f"  Win rate: {win_rate*100:.2f}%")
        print(f"  Avg trade P&L: {avg_trade_pnl:.4f}")
        print(f"  Position usage: {pos_usage}")

        if operator.logger:
            operator.logger.log("operator", "episode_metrics", {
                "pair": f"{pair[0]}-{pair[1]}",
                **extra_stats,
                "sharpe": sharpe,
                "sortino": sortino
            })

        if len(episode_traces) > 0:
            final_cum_return = episode_traces[-1].get('cum_return', 0)
            final_pnl = episode_traces[-1].get('realized_pnl', 0)
            print(f"  Final return: {final_cum_return*100:.2f}%")
            print(f"  Total P&L: {final_pnl:.2f}")

            if operator.logger:
                operator.logger.log("operator", "episode_complete", {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "total_steps": len(episode_traces),
                    "final_cum_return": final_cum_return,
                    "total_pnl": final_pnl,
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "was_skipped": skip_to_next_pair
                })

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    print("\n" + "="*70)
    print("HOLDOUT TESTING COMPLETE")
    print("="*70)
    print(f"Total steps: {global_step}")
    print(f"Total pairs tested: {len(pairs)}")
    print(f"Pairs completed: {len(pairs) - len(skipped_pairs)}")
    print(f"Pairs skipped by supervisor: {len(skipped_pairs)}")
    print("="*70)
    
    # Print skipped pairs summary
    if skipped_pairs:
        print(f"\n{'='*70}")
        print(f"SUPERVISOR INTERVENTION SUMMARY")
        print(f"{'='*70}")
        print(f"{len(skipped_pairs)} pairs stopped early:\n")
        
        for skip in skipped_pairs:
            print(f"  {skip['pair']}:")
            print(f"    Severity: {skip.get('severity', 'unknown').upper()}")
            print(f"    Reason: {skip['reason']}")
            print(f"    Stopped at local step: {skip['local_step_stopped']}")
            metrics = skip.get('metrics', {})
            print(f"    Final metrics: DD={metrics.get('drawdown', 0):.2%}, "
                  f"Sharpe={metrics.get('sharpe', 0):.2f}, "
                  f"WinRate={metrics.get('win_rate', 0):.2%}\n")
        print("="*70)

    return all_traces, skipped_pairs

def calculate_sharpe(traces, risk_free_rate=None):
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    
    returns = np.array([t['daily_return'] for t in traces if t['daily_return'] != 0])
    
    if len(returns) < 2:
        return 0.0
    
    rf_daily = risk_free_rate / 252
    excess_returns = returns - rf_daily
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess < 1e-8:
        return 0.0
    
    return (mean_excess / std_excess) * np.sqrt(252)


def calculate_sortino(traces, risk_free_rate=None):
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    
    returns = np.array([t['daily_return'] for t in traces if t['daily_return'] != 0])
    
    if len(returns) < 2:
        return 0.0
    
    rf_daily = risk_free_rate / 252
    excess_returns = returns - rf_daily
    
    mean_excess = np.mean(excess_returns)
    downside_deviation = np.sqrt(np.mean(np.minimum(0, excess_returns)**2))
    
    if downside_deviation < 1e-8:
        return 100.0 if mean_excess > 0 else 0.0
    
    return (mean_excess / downside_deviation) * np.sqrt(252)
    
def save_detailed_trace(self, trace: Dict[str, Any], filepath: str = "traces/operator_detailed.json"):
    """Save detailed trace for visualization."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "a") as f:
        f.write(json.dumps(trace, default=str) + "\n")
