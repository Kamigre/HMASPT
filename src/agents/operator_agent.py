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

# Ensure config is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import CONFIG
from agents.message_bus import JSONLogger


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
        
        # Observation space: 14 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
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
        
        # Volatility Ratio
        self.vol_ratio = self.vol_short / (self.vol_long + 1e-8)
        
        # 3. Momentum Features (Trend Detection)
        self.rsi = self._compute_rsi(self.spread, period=14)
        
        # Convert to numpy and fill NaNs
        self.spread_np = np.nan_to_num(self.spread.to_numpy(), nan=0.0)
        self.zscore_short_np = np.nan_to_num(self.zscore_short.to_numpy(), nan=0.0)
        self.zscore_long_np = np.nan_to_num(self.zscore_long.to_numpy(), nan=0.0)
        self.vol_np = np.nan_to_num(self.vol_short.to_numpy(), nan=1.0)
        self.vol_ratio_np = np.nan_to_num(self.vol_ratio.to_numpy(), nan=1.0)
        self.rsi_np = np.nan_to_num(self.rsi.to_numpy(), nan=50.0)
        
        # Store prices for logging
        self.price_x_np = x.to_numpy()
        self.price_y_np = y.to_numpy()

    def _get_observation(self, idx: int) -> np.ndarray:
        """Build NORMALIZED observation vector"""
        if idx < 0 or idx >= len(self.spread_np):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        norm_unrealized = self.unrealized_pnl / self.initial_capital
        norm_realized = self.realized_pnl / self.initial_capital
        
        obs = np.array([
            self.zscore_short_np[idx],
            self.zscore_long_np[idx],
            self.vol_np[idx],
            self.spread_np[idx],
            
            # NEW FEATURES
            self.rsi_np[idx] / 100.0,
            self.vol_ratio_np[idx],
            
            float(self.position / self.position_scale),  
            float(self.entry_spread) if self.position != 0 else 0.0,
            
            # NORMALIZED FINANCIALS
            float(norm_unrealized),
            float(norm_realized),
            
            float(self.cash / self.initial_capital - 1),  
            float(self.portfolio_value / self.initial_capital - 1),  
            
            float(self.days_in_position) / 252.0,
            float(self.num_trades) / 100.0,
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
        Execute one trading step with IMPROVED Reward Calculation.
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
        current_zscore = float(self.zscore_short_np[current_idx])
        
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
                
                # Check if we are closing or flipping
                if target_position == 0 or np.sign(target_position) != np.sign(self.position):
                    closed_size = abs(self.position)
                else:
                    closed_size = abs(position_change)
                    
                # Standard PnL Calculation
                realized_pnl_this_step = (self.position / abs(self.position)) * closed_size * spread_change

            # Transaction Costs
            trade_size = abs(position_change)
            notional = trade_size * abs(current_spread)
            transaction_costs = notional * self.transaction_cost_rate
            self.num_trades += 1
            
            # Reset/Update Entry Price
            if target_position != 0 and np.sign(target_position) != np.sign(self.position):
                # Flipping or Opening New
                self.entry_spread = current_spread
                self.days_in_position = 0
            elif target_position == 0:
                # Flat
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
            # Holding existing position
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

        # Calculate log returns for better stability, or standard % returns
        daily_return = (self.portfolio_value - self.prev_portfolio_value) / max(self.prev_portfolio_value, 1e-8)
        self.prev_portfolio_value = self.portfolio_value

        # 6. Metrics
        prev_peak = self.peak_value
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8)
        
                # ===============================
        # 7. SIMPLIFIED REWARD FUNCTION
        # ===============================
        
        reward = 0.0
        
        # --- Return-based reward -----------------------------------
        # Risk-adjusted return: reward profits, penalize losses slightly more
        if daily_return >= 0:
            reward += daily_return
        else:
            reward += 1.2 * daily_return   # loss aversion
        
        # --- Z-score alignment (tiny shaping) ------------------------
        norm_pos = self.position / self.position_scale
        
        # anti-alignment = penalize
        if (current_zscore > 1 and norm_pos > 0) or (current_zscore < -1 and norm_pos < 0):
            reward -= 0.25
        
        # pro-alignment = encourage
        if (current_zscore > 1 and norm_pos < 0) or (current_zscore < -1 and norm_pos > 0):
            reward += 0.25
        
        # --- Clip for stability --------------------------------------
        reward = float(np.clip(reward, -1.0, 1.0))
        
        # D) Clip for Stability (Crucial for outliers like 0.33)
        # ------------------------------------------------------------------
        # tanh compresses the massive 33% return (Value 16.5) to 1.0
        # while keeping the 1% return (Value 0.5) at ~0.46 (linear).
        reward = float(np.tanh(reward))
        
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
            'z_score': float(current_zscore), 
            'days_in_position': int(self.days_in_position),
            'daily_return': float(daily_return),
            'drawdown': float(drawdown),
            'num_trades': int(self.num_trades),
            'trade_occurred': bool(trade_occurred),
            'cum_return': float(self.portfolio_value / self.initial_capital - 1),
            'forced_close': is_last_step and trade_occurred,
            'price_x': float(self.price_x_np[current_idx]),
            'price_y': float(self.price_y_np[current_idx])
        }
        
        terminated = is_last_step
        
        return obs, float(reward), terminated, False, info
        
@dataclass
class OperatorAgent:
    
    logger: Optional[JSONLogger] = None
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
            learning_rate=0.001,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.04,
            verbose=1,
            device="auto",
            seed=seed,
            policy_kwargs=policy_kwargs 
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

        # Calculate metrics using homogenous methods
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

    def save_detailed_trace(self, trace: Dict[str, Any], filepath: str = "traces/operator_detailed.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a") as f:
            f.write(json.dumps(trace, default=str) + "\n")


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


def run_operator_holdout(operator, holdout_prices, pairs, supervisor, warmup_steps=90):
    """
    Run holdout testing with supervisor monitoring.
    Features FORCE LIQUIDATION on stop, ensuring metrics reflect the exit costs.
    """
    
    # Check supervisor config
    if "supervisor_rules" in CONFIG and "holdout" in CONFIG["supervisor_rules"]:
        check_interval = CONFIG["supervisor_rules"]["holdout"].get("check_interval", 20)
    else:
        check_interval = 20
        
    operator.traces_buffer = []
    operator.current_step = 0

    global_step = 0
    all_traces = []
    skipped_pairs = []
    pair_summaries = []

    lookback = CONFIG.get("rl_lookback", 30)
    
    for pair in pairs:
        print(f"\n{'='*70}")
        print(f"Testing pair: {pair[0]} - {pair[1]}")
        print(f"{'='*70}")

        # 1. Data Validation
        if pair[0] not in holdout_prices.columns or pair[1] not in holdout_prices.columns:
            print(f"⚠️ Warning: Tickers {pair} not found in holdout data - skipping")
            continue

        series_x = holdout_prices[pair[0]].dropna()
        series_y = holdout_prices[pair[1]].dropna()
        aligned = pd.concat([series_x, series_y], axis=1).dropna()

        if len(aligned) < lookback + warmup_steps + 1:
            print(f"⚠️ Insufficient data ({len(aligned)} steps). Needs {lookback + warmup_steps + 1}")
            continue

        # 2. Model Loading
        model_path = os.path.join(operator.storage_dir, f"operator_model_{pair[0]}_{pair[1]}.zip")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found - skipping")
            continue

        model = operator.load_model(model_path)

        # 3. Environment Setup
        env = PairTradingEnv(
            series_x=aligned.iloc[:, 0], 
            series_y=aligned.iloc[:, 1], 
            lookback=lookback, 
            initial_capital=10000,
            transaction_cost_rate=0.0005, 
            test_mode=True
        )

        episode_traces = []
        local_step = 0
        obs, info = env.reset() 

        # --- WARM-UP PHASE ---
        print(f" ⏳ Warming up model state on {warmup_steps} steps...")
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        warmup_completed = True
        
        for i in range(warmup_steps):
            if env.idx >= len(env.spread_np) - 1:
                warmup_completed = False
                break 
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            episode_starts = np.array([done])
            if done: warmup_completed = False; break
        
        if not warmup_completed: continue

        # --- FINANCIAL RESET ---
        env.cash = env.initial_capital
        env.portfolio_value = env.initial_capital
        env.realized_pnl = 0.0 
        env.unrealized_pnl = 0.0 
        env.num_trades = 0
        env.trade_history = []
        env.peak_value = env.initial_capital
        if env.position != 0:
            env.position = 0
            env.entry_spread = 0.0
            env.days_in_position = 0

        # --- MAIN TRADING LOOP ---
        terminated = False
        stop_triggered = False
        intervention_reason = ""
        intervention_severity = ""
        
        while not terminated:
            
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)
            episode_starts = np.array([terminated])

            # Standard Trace
            trace = {
                "pair": f"{pair[0]}-{pair[1]}",
                "step": global_step,
                "local_step": local_step,
                "portfolio_value": float(info.get("portfolio_value", 0.0)),
                "cum_return": float(info.get("cum_return", 0.0)),
                "position": float(info.get("position", 0)),
                "max_drawdown": float(info.get("drawdown", 0)),
                
                # --- PnL METRICS ---
                "realized_pnl": float(info.get("realized_pnl", 0.0)),           # <--- ADDED: Accumulated PnL
                "realized_pnl_this_step": float(info.get("realized_pnl_this_step", 0.0)),
                
                "transaction_costs": float(info.get("transaction_costs", 0.0)),
                "daily_return": float(info.get("daily_return", 0.0)),
                "current_spread": float(info.get("current_spread", 0.0)),
                "z_score": float(info.get("z_score", 0.0)), 
                "days_in_position": int(info.get("days_in_position", 0)),
                # Log raw prices for behavior analysis
                "price_x": float(info.get("price_x", 0.0)),
                "price_y": float(info.get("price_y", 0.0))
            }

            episode_traces.append(trace)
            all_traces.append(trace)
            operator.add_trace(trace)

            # --- SUPERVISOR MONITORING ---
            if local_step > 0 and local_step % check_interval == 0:
                decision = supervisor.check_operator_performance(episode_traces, pair, phase="holdout")
                
                if decision["action"] == "stop":
                    intervention_severity = decision.get("severity", "critical")
                    intervention_reason = decision['reason']
                    print(f"\n⛔ SUPERVISOR INTERVENTION [{intervention_severity.upper()}]: Stopping pair early")
                    print(f"    Reason: {intervention_reason}")

                    # ========================================================
                    # FORCE LIQUIDATION: REALIZE THE LOSS/PROFIT
                    # ========================================================
                    if env.position != 0:
                        print(f"    ⚠️ Force Closing open position ({env.position}) to realize PnL...")
                        
                        # Step environment with Action 1 (Flat)
                        # We must update the env state so the final portfolio value reflects the close
                        obs, reward, terminated, _, info = env.step(1)
                        
                        forced_pnl = info.get('realized_pnl_this_step', 0.0)
                        forced_cost = info.get('transaction_costs', 0.0)
                        
                        print(f"    💸 Liquidation Result: PnL {forced_pnl:.2f}, Costs {forced_cost:.4f}")
                        
                        # Create final trace reflecting the close
                        final_trace = trace.copy()
                        final_trace['step'] += 1
                        final_trace['local_step'] += 1
                        final_trace['position'] = 0 # Flat
                        final_trace['realized_pnl_this_step'] = forced_pnl
                        final_trace['transaction_costs'] = forced_cost
                        final_trace['portfolio_value'] = float(info.get("portfolio_value", trace['portfolio_value']))
                        
                        # --- UPDATE ACCUMULATED PnL IN FINAL TRACE ---
                        final_trace['realized_pnl'] = float(info.get("realized_pnl", trace['realized_pnl'])) # <--- ADDED
                        
                        final_trace['cum_return'] = float(info.get("cum_return", trace['cum_return']))
                        final_trace['max_drawdown'] = float(info.get("drawdown", trace['max_drawdown']))
                        final_trace['daily_return'] = float(info.get("daily_return", 0.0))
                        
                        # Append this trace so metrics include the forced close
                        episode_traces.append(final_trace)
                        all_traces.append(final_trace)
                        operator.add_trace(final_trace)

                    stop_triggered = True
                    break # Exit loop
                
                elif decision["action"] == "adjust":
                    print(f"\n⚠️  SUPERVISOR WARNING: {decision['reason']}")

            local_step += 1
            global_step += 1
            operator.current_step = global_step

        # --- METRICS REPORTING (Including Forced Close) ---
        if len(episode_traces) > 0:
            # 1. Recalculate metrics on the FULL trace history (including liquidation)
            sharpe = calculate_sharpe(episode_traces)
            sortino = calculate_sortino(episode_traces)
            final_return = episode_traces[-1]['cum_return'] * 100
            max_dd = episode_traces[-1]['max_drawdown']
            
            # CONSISTENT WIN RATE LOGIC: NET PNL > 0
            pnl_events = [t for t in episode_traces if t['realized_pnl_this_step'] != 0]
            if len(pnl_events) > 0:
                wins = len([t for t in pnl_events if (t['realized_pnl_this_step'] - t['transaction_costs']) > 0])
                win_rate = (wins / len(pnl_events)) * 100
            else:
                win_rate = 0.0

            # --- [NEW] LOGGING METRICS ---
            # Explicitly log the calculated metrics to JSONLogger
            holdout_metrics = {
                "pair": f"{pair[0]}-{pair[1]}",
                "final_return": final_return,
                "max_drawdown": max_dd,
                "sharpe": sharpe,
                "sortino": sortino,
                "win_rate": win_rate,
                "total_trades": len(pnl_events),
                "steps": local_step,
                "status": "STOPPED" if stop_triggered else "COMPLETE",
                "intervention_reason": intervention_reason if stop_triggered else None
            }
            
            if operator.logger:
                operator.logger.log("operator", "holdout_complete", holdout_metrics)
            # -----------------------------

            # 2. Log skipped pair info with the FINAL metrics
            if stop_triggered:
                skip_info = {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "reason": intervention_reason,
                    "severity": intervention_severity,
                    "step_stopped": global_step,
                    "final_return": final_return,
                    "sharpe": sharpe,
                    "drawdown": max_dd
                }
                skipped_pairs.append(skip_info)

            # 3. Print Detailed Results
            status_str = f"⛔ STOPPED ({intervention_reason})" if stop_triggered else "✅ COMPLETE"
            
            print(f"\n📊 Holdout Results for {pair[0]}-{pair[1]}:")
            print(f"  Status:         {status_str}")
            print(f"  Final Return:   {final_return:.2f}%")
            print(f"  Max Drawdown:   {max_dd:.2%}")
            print(f"  Sharpe Ratio:   {sharpe:.3f}")
            print(f"  Sortino Ratio:  {sortino:.3f}")
            print(f"  Win Rate:       {win_rate:.1f}% ({len(pnl_events)} trades)")
            print(f"  Accum. PnL:     {episode_traces[-1]['realized_pnl']:.2f}") # <--- Optional: Print it for verification
            
            # Position Distribution
            positions = [t['position'] for t in episode_traces]
            print(f"  Position Distribution:")
            for pos, name in [(100, "Long"), (-100, "Short"), (0, "Flat")]:
                count = np.sum(np.array(positions) == pos)
                pct = count / len(positions) * 100
                print(f"    {name:<5}: {pct:.1f}%")

            pair_summaries.append({
                "pair": f"{pair[0]}-{pair[1]}",
                "return": final_return,
                "sharpe": sharpe,
                "drawdown": max_dd,
                "win_rate": win_rate,
                "status": "STOPPED" if stop_triggered else "COMPLETE"
            })
            
    # Final Summary Table
    print("\n" + "="*90)
    print("HOLDOUT TESTING COMPLETE: SUMMARY")
    print("="*90)
    print(f"{'Pair':<15} | {'Status':<9} | {'Return':<8} | {'Sharpe':<6} | {'Sortino':<7} | {'Max DD':<8} | {'Win Rate':<8}")
    print("-" * 90)
    
    total_ret = 0
    for s in pair_summaries:
        status_icon = "🛑" if s['status'] == "STOPPED" else "✅"
        print(f"{s['pair']:<15} | {status_icon} {s['status'][:3]}.. | {s['return']:>7.2f}% | {s['sharpe']:>6.2f} | {s['sharpe']:>7.2f} | {s['drawdown']:>7.1%} | {s['win_rate']:>7.1f}%")
        total_ret += s['return']
        
    avg_ret = total_ret / len(pair_summaries) if pair_summaries else 0.0
    print("-" * 90)
    print(f"Average Return: {avg_ret:.2f}% across {len(pair_summaries)} pairs")
    print("="*90)
    
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
    
    if len(downside_returns) == 0:
        return 0.0
        
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation < 1e-9: return 0.0      
    return (mean_excess / downside_deviation) * np.sqrt(252)
