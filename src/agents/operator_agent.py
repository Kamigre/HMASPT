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
from sb3_contrib import RecurrentPPO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

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
        
        # Observation space INCREASED to 14 (Added RSI and Vol Ratio)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        # PARAMETERS
        self.reward_scale = 10.0
        self.drawdown_penalty_factor = 0.5
        self.holding_penalty = 0.05
        self.profit_bonus = 2.0
        
        # Precompute spread and features
        self._precompute_features()
        
        self.reset()

    def _compute_rsi(self, series, period=14):
        """Helper to calculate RSI of the spread"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _precompute_features(self):
        """Compute spread and advanced features"""
        x = self.data.iloc[:, 0]
        y = self.data.iloc[:, 1]
        
        # Raw spread
        self.spread = x - y
        
        # 1. Z-scores (Mean Reversion Signals)
        self.zscore_short = (
            (self.spread - self.spread.rolling(self.lookback).mean()) / 
            (self.spread.rolling(self.lookback).std() + 1e-8)
        )
        
        self.zscore_long = (
            (self.spread - self.spread.rolling(self.lookback * 2).mean()) / 
            (self.spread.rolling(self.lookback * 2).std() + 1e-8)
        )
        
        # 2. Volatility Features (Risk Detection)
        self.vol_short = self.spread.rolling(self.lookback).std()
        self.vol_long = self.spread.rolling(self.lookback * 3).std()
        
        # Volatility Ratio: If > 1.0, market is becoming more turbulent (Regime Change)
        self.vol_ratio = self.vol_short / (self.vol_long + 1e-8)
        
        # 3. Momentum Features (Trend Detection)
        # RSI helps distinguish "Oversold" (Good Entry) from "Crashing" (Bad Entry)
        self.rsi = self._compute_rsi(self.spread, period=14)
        
        # Convert to numpy and fill NaNs
        self.spread_np = np.nan_to_num(self.spread.to_numpy(), nan=0.0)
        self.zscore_short_np = np.nan_to_num(self.zscore_short.to_numpy(), nan=0.0)
        self.zscore_long_np = np.nan_to_num(self.zscore_long.to_numpy(), nan=0.0)
        self.vol_np = np.nan_to_num(self.vol_short.to_numpy(), nan=1.0)
        self.vol_ratio_np = np.nan_to_num(self.vol_ratio.to_numpy(), nan=1.0)
        self.rsi_np = np.nan_to_num(self.rsi.to_numpy(), nan=50.0) # Default to neutral 50

    def _get_observation(self, idx: int) -> np.ndarray:
        """Build NORMALIZED observation vector"""
        if idx < 0 or idx >= len(self.spread_np):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Normalize financial metrics by initial capital
        # This keeps values generally between -1.0 and 1.0, which neural nets love.
        norm_unrealized = self.unrealized_pnl / self.initial_capital
        norm_realized = self.realized_pnl / self.initial_capital
        
        obs = np.array([
            self.zscore_short_np[idx],
            self.zscore_long_np[idx],
            self.vol_np[idx],
            self.spread_np[idx],
            
            # NEW FEATURES
            self.rsi_np[idx] / 100.0,       # Scale RSI to 0.0-1.0
            self.vol_ratio_np[idx],         # Ratio is already small (~1.0)
            
            float(self.position / self.position_scale),  
            float(self.entry_spread) if self.position != 0 else 0.0,
            
            # NORMALIZED FINANCIALS
            float(norm_unrealized),
            float(norm_realized),
            
            float(self.cash / self.initial_capital - 1),  
            float(self.portfolio_value / self.initial_capital - 1),  
            
            float(self.days_in_position) / 252.0, # Scale days to ~0.0-1.0 (assuming max 1 year hold)
            float(self.num_trades) / 100.0,       # Soft scaling for trade count
        ], dtype=np.float32)
        
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.idx = self.lookback if not self.test_mode else 0
        self.position = 0
        self.entry_spread = 0.0 
        self.days_in_position = 0
        
        # Financial tracking
        self.cash = self.initial_capital
        self.realized_pnl = 0.0 
        self.unrealized_pnl = 0.0 
        self.portfolio_value = self.initial_capital
        
        # Performance tracking
        self.peak_value = self.initial_capital
        self.num_trades = 0
        self.trade_history = []
        
        # For return calculation
        self.prev_portfolio_value = self.initial_capital
        
        return self._get_observation(self.idx), {}

    def step(self, action: int):
        """
        Execute one trading step. 
        Forces a close (position=0) at the last timestep.
        """
        current_idx = self.idx
        
        # 1. Determine if this is the last available step
        is_last_step = (current_idx >= len(self.spread_np) - 1)
        
        # 2. Determine Action
        if is_last_step:
            target_position = 0 # FORCE EXIT
        else:
            base_position = int(action) - 1
            target_position = base_position * self.position_scale

        # 3. Setup Data
        current_spread = float(self.spread_np[current_idx])
        
        if is_last_step:
            next_spread = current_spread 
            next_idx = current_idx 
        else:
            next_idx = current_idx + 1
            next_spread = float(self.spread_np[next_idx])
            
        # 4. Execute Trade & Update Financials
        position_change = target_position - self.position
        trade_occurred = (position_change != 0)
        
        realized_pnl_this_step = 0.0
        transaction_costs = 0.0
        
        if trade_occurred:
            # Calculate Realized P&L
            if self.position != 0:
                spread_change = current_spread - self.entry_spread
                
                if target_position == 0 or np.sign(target_position) != np.sign(self.position):
                    closed_size = abs(self.position)
                else:
                    closed_size = abs(position_change)
                    
                realized_pnl_this_step = (self.position / abs(self.position)) * closed_size * spread_change

            # Costs
            trade_size = abs(position_change)
            notional = trade_size * abs(current_spread)
            transaction_costs = notional * self.transaction_cost_rate
            self.num_trades += 1
            
            # Reset Entry Price
            if target_position != 0 and np.sign(target_position) != np.sign(self.position):
                self.entry_spread = current_spread
                self.days_in_position = 0
            elif target_position == 0:
                self.entry_spread = 0.0
                self.days_in_position = 0
                
            # Log history
            if self.position != 0:
                 self.trade_history.append({
                    'entry_spread': self.entry_spread,
                    'exit_spread': current_spread,
                    'position': self.position,
                    'pnl': realized_pnl_this_step,
                    'holding_days': self.days_in_position,
                    'forced_close': is_last_step
                })
        else:
            self.days_in_position += 1
            
        # Update State
        self.position = target_position
        self.realized_pnl += realized_pnl_this_step - transaction_costs
        self.cash = self.initial_capital + self.realized_pnl
        
        if self.position != 0:
            self.unrealized_pnl = self.position * (next_spread - self.entry_spread)
        else:
            self.unrealized_pnl = 0.0
            
        self.portfolio_value = self.cash + self.unrealized_pnl
        
        # 5. Returns
        if not hasattr(self, 'prev_portfolio_value'):
            self.prev_portfolio_value = self.initial_capital

        daily_return = (self.portfolio_value - self.prev_portfolio_value) / max(self.prev_portfolio_value, 1e-8)
        self.prev_portfolio_value = self.portfolio_value
        
        # 6. Metrics
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8)
        
        # 7. Reward
        reward = daily_return * 100.0
        reward -= 0.5 * drawdown
        if self.position != 0:
            reward -= 0.05 * self.days_in_position
        if realized_pnl_this_step > 0:
            reward += 2.0 * (realized_pnl_this_step / self.initial_capital) * 100.0
        reward = np.clip(reward, -10.0, 10.0)
        
        # 8. Index
        if not is_last_step:
            self.idx = next_idx
        
        # 9. Obs
        obs = self._get_observation(self.idx)
        
        # 10. Info
        info = {
            'portfolio_value': float(self.portfolio_value),
            'cash': float(self.cash),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl_this_step': float(realized_pnl_this_step),
            'transaction_costs': float(transaction_costs),
            'position': int(self.position),
            'entry_spread': float(self.entry_spread),
            'current_spread': float(current_spread),
            'days_in_position': int(self.days_in_position),
            'daily_return': float(daily_return),
            'drawdown': float(drawdown),
            'num_trades': int(self.num_trades),
            'trade_occurred': bool(trade_occurred),
            'cum_return': float(self.portfolio_value / self.initial_capital - 1),
            'forced_close': is_last_step and trade_occurred
        }
        
        terminated = is_last_step
        
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
        return RecurrentPPO.load(model_path)

    def train_on_pair(self, prices: pd.DataFrame, x: str, y: str,
                      lookback: int = None, timesteps: int = None,
                      shock_prob: float = None, shock_scale: float = None,
                      use_curriculum: bool = False):

        # Get seed from CONFIG
        seed = CONFIG.get("random_seed", 42)
                           
        if not self.active:
            return None

        if lookback is None:
            lookback = CONFIG.get("rl_lookback", 30) 
            
        if timesteps is None:
            timesteps = CONFIG.get("rl_timesteps", 500000)

        series_x = prices[x]
        series_y = prices[y]

        print(f"\n{'='*70}")
        print(f"Training pair: {x} - {y} (LSTM POLICY)")
        print(f"  Data length: {len(series_x)} days")
        print(f"  Timesteps: {timesteps:,}")
        print(f"  Time Window (Lookback): {lookback} (Paper optimal: 30)")
        print(f"  LSTM Hidden Size: 512 (Paper optimal)")
        print(f"{'='*70}")

        print("\n🚀 Training with Recurrent PPO (LSTM)...")
        env = PairTradingEnv(
            series_x, series_y, lookback, position_scale=100,
            transaction_cost_rate=0.0005, test_mode=False
        )
        
        # Seed the environment
        env.reset(seed=seed)

        policy_kwargs = dict(
            lstm_hidden_size=512,
            n_lstm_layers=1
        )

        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=0.0001,
            n_steps=4096,
            batch_size=256,
            n_epochs=20,
            gamma=0.99,
            # 0.01 is standard. 0.05 forces the agent to try random actions 
            # (buying/selling) more often during early training.
            ent_coef=0.05,
            verbose=1,
            device="auto",
            seed=seed,
            policy_kwargs=policy_kwargs  # Applied hyperparams
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

        # Initialize LSTM states
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            obs, reward, done, _, info = env_eval.step(action)
            episode_starts = np.array([done])
            
            daily_returns.append(info.get('daily_return', 0))
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

        # PAPER IMPLEMENTATION: Ensure test environment uses same lookback (30)
        lookback = CONFIG.get("rl_lookback", 30)

        # TEST ENVIRONMENT
        env = PairTradingEnv(
              series_x=aligned.iloc[:, 0], series_y=aligned.iloc[:, 1], 
              lookback=lookback, position_scale=100,
              transaction_cost_rate = 0.0005, test_mode=True
          )

        episode_traces = []
        local_step = 0
        obs, info = env.reset()
        terminated = False
        skip_to_next_pair = False

        # Initialize LSTM states
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        # Trading loop with supervisor monitoring
        while not terminated and not skip_to_next_pair:
            # Pass state and episode_start to model
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            
            obs, reward, terminated, _, info = env.step(action)
            episode_starts = np.array([terminated])

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
                    
                    # ---------------------------------------------------------
                    # FORCE CLOSE POSITION LOGIC ADDED HERE
                    # ---------------------------------------------------------
                    if info.get('position', 0) != 0:
                        print(f"   ⚠️ Force closing position {info.get('position')} to realize PnL...")
                        
                        # Action 1 = Flat/Close (since base_position = action - 1)
                        # Action 0 -> Short (-1), Action 1 -> Flat (0), Action 2 -> Long (1)
                        action_close = 1 
                        
                        # Execute the closing step
                        obs, reward, terminated, _, info = env.step(action_close)
                        
                        # Create trace for the closing action
                        closing_trace = {
                            "pair": f"{pair[0]}-{pair[1]}",
                            "step": global_step + 1,
                            "local_step": local_step + 1,
                            "reward": float(reward),
                            "portfolio_value": float(info.get("portfolio_value", 0.0)),
                            "cum_return": float(info.get("cum_return", 0.0)),
                            "cum_reward": float(info.get("cum_reward", 0.0)),
                            "position": float(info.get("position", 0)), # Should be 0 now
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
                            "forced_close_by_supervisor": True
                        }
                        
                        episode_traces.append(closing_trace)
                        all_traces.append(closing_trace)
                        operator.add_trace(closing_trace)
                        
                        if hasattr(operator, 'save_detailed_trace'):
                            operator.save_detailed_trace(closing_trace)
                            
                        # Increment global steps to account for the closing action
                        global_step += 1
                        local_step += 1

                    # Record skip information
                    skip_info = {
                        "pair": f"{pair[0]}-{pair[1]}",
                        "reason": decision['reason'],
                        "severity": severity,
                        "step_stopped": global_step,
                        "local_step_stopped": local_step,
                        "metrics": decision['metrics'],
                        "final_pnl": float(info.get("realized_pnl", 0.0)) # PnL after close
                    }
                    
                    skipped_pairs.append(skip_info)
                    skip_to_next_pair = True
                    
                    if operator.logger:
                        operator.logger.log("supervisor", "intervention", skip_info)
                    
                    continue
                
                elif decision["action"] == "adjust":
                    print(f"\n⚠️  SUPERVISOR WARNING [{decision.get('severity', 'warning').upper()}]:")
                    print(f"   {decision['reason']}")
                    if 'suggestion' in decision:
                        print(f"   💡 Suggestion: {decision['suggestion']}")
                
                elif decision["action"] == "warn":
                    if local_step % (check_interval * 4) == 0:
                        print(f"\nℹ️  SUPERVISOR INFO:")
                        print(f"   {decision['reason']}")
                
                if local_step % (check_interval * 2) == 0:
                    metrics = decision["metrics"]
                    print(f"\n📊 Step {local_step}: DD={metrics.get('drawdown', 0):.2%}, "
                          f"Sharpe={metrics.get('sharpe', 0):.2f}")

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
        positions = np.array([t["position"] for t in episode_traces])

        # Trade detection
        trades = []
        last_position = 0
        for t in episode_traces:
            pos = t["position"]
            pnl = t.get("realized_pnl_this_step", 0)
            if pos != last_position and pnl != 0:
                trades.append({"position": pos, "realized_pnl": pnl})
            last_position = pos

        pnls_list = [tr["realized_pnl"] for tr in trades]
        n_trades = len(pnls_list)
        wins = [1 for pnl in pnls_list if pnl > 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_trade_pnl = np.mean(pnls_list) if n_trades > 0 else 0.0

        # Position usage
        unique_positions, pos_counts = np.unique(positions, return_counts=True)
        pos_usage = {
            int(p): float(c) / len(positions)
            for p, c in zip(unique_positions, pos_counts)
        }

        # Return metrics
        filtered_returns = [t.get("daily_return", 0) for t in episode_traces if t.get("daily_return", 0) != 0]
        ret_mean = np.mean(filtered_returns) if filtered_returns else 0.0
        ret_std = np.std(filtered_returns) if filtered_returns else 0.0
        ret_median = np.median(filtered_returns) if filtered_returns else 0.0
        max_dd = max(t.get("max_drawdown", 0) for t in episode_traces) if episode_traces else 0

        # Sharpe and Sortino
        sharpe = calculate_sharpe(episode_traces)
        sortino = calculate_sortino(episode_traces)

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

    print("\n" + "="*70)
    print("HOLDOUT TESTING COMPLETE")
    print("="*70)
    print(f"Total steps: {global_step}")
    print(f"Total pairs tested: {len(pairs)}")
    print(f"Pairs completed: {len(pairs) - len(skipped_pairs)}")
    print(f"Pairs skipped by supervisor: {len(skipped_pairs)}")
    print("="*70)
    
    if skipped_pairs:
        print(f"\n{'='*70}")
        print(f"SUPERVISOR INTERVENTION SUMMARY")
        print(f"{'='*70}")
        for skip in skipped_pairs:
            print(f"  {skip['pair']}: {skip['reason']} (Metric: {skip['metrics']})")
        print("="*70)

    return all_traces, skipped_pairs

def calculate_sharpe(traces, risk_free_rate=None):
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    returns = np.array([t.get('daily_return', 0.0) for t in traces])
    if len(returns) < 2: return 0.0
    rf_daily = risk_free_rate / 252.0
    excess_returns = returns - rf_daily
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    if std_excess < 1e-9: return 0.0
    return (mean_excess / std_excess) * np.sqrt(252)

def calculate_sortino(traces, risk_free_rate=None):
    if risk_free_rate is None:
        risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    returns = np.array([t.get('daily_return', 0.0) for t in traces])
    if len(returns) < 2: return 0.0
    rf_daily = risk_free_rate / 252.0
    excess_returns = returns - rf_daily
    mean_excess = np.mean(excess_returns)
    
    # Sortino uses downside deviation of excess returns below 0
    downside_returns = excess_returns[excess_returns < 0]
    
    # If no downside, Sortino is infinite (technically), but we return high value or 0
    if len(downside_returns) == 0:
        return 0.0
        
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation < 1e-9: return 0.0    
    return (mean_excess / downside_deviation) * np.sqrt(252)
    
def save_detailed_trace(self, trace: Dict[str, Any], filepath: str = "traces/operator_detailed.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(trace, default=str) + "\n")
