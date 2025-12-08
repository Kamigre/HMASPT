import os
import json
import time
import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import CONFIG 
from utils import half_life, compute_spread
from agents.message_bus import JSONLogger

def calculate_sharpe(traces, risk_free_rate=None):
    if risk_free_rate is None: risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    returns = np.array([t['daily_return'] for t in traces if t['daily_return'] != 0])
    if len(returns) < 2: return 0.0
    rf_daily = risk_free_rate / 252
    excess_returns = returns - rf_daily
    mean_exc = np.mean(excess_returns)
    std_exc = np.std(excess_returns, ddof=1)
    if std_exc < 1e-8: return 0.0
    return (mean_exc / std_exc) * np.sqrt(252)

def calculate_sortino(traces, risk_free_rate=None):
    if risk_free_rate is None: risk_free_rate = CONFIG.get("risk_free_rate", 0.04)
    returns = np.array([t['daily_return'] for t in traces if t['daily_return'] != 0])
    if len(returns) < 2: return 0.0
    rf_daily = risk_free_rate / 252
    excess_returns = returns - rf_daily
    mean_exc = np.mean(excess_returns)
    downside_deviation = np.sqrt(np.mean(np.minimum(0, excess_returns)**2))
    if downside_deviation < 1e-8: return 100.0 if mean_exc > 0 else 0.0
    return (mean_exc / downside_deviation) * np.sqrt(252)

def save_detailed_trace(self, trace: Dict[str, Any], filepath: str = "traces/operator_detailed.json"):
    # Attempt to use the operator's storage_dir if available
    base_dir = getattr(self, 'storage_dir', os.path.dirname(filepath))
    full_path = os.path.join(base_dir, filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "a") as f:
        f.write(json.dumps(trace, default=str) + "\n")


# ===================================================================
# 2. PAIR TRADING ENVIRONMENT (RISK-ADJUSTED, CONTINUOUS V3)
# ===================================================================

class PairTradingEnv(gym.Env):

    def __init__(self, series_x: pd.Series, series_y: pd.Series,
                 lookback: int = 30,
                 initial_capital: float = 10000,
                 transaction_cost_rate: float = 0.0005,
                 test_mode: bool = False):
        
        super().__init__()
        
        # Align series
        self.data = pd.concat([series_x, series_y], axis=1).dropna()
        self.lookback = lookback
        self.test_mode = test_mode
        self.initial_capital = initial_capital
        self.transaction_cost_rate = transaction_cost_rate
        
        # --- CONTINUOUS ACTION PARAMETERS (Maximum Unit Size for Normalization/Limit) ---
        self.MAX_POSITION_CAP = 0.50 # Max 50% of initial capital (Notional/Capital)
        self.MAX_POSITION_UNITS = 1000 # Max units for scaling purposes (adjust if needed)
        
        # --- RISK PARAMETERS (Hard Stop-Loss/Take-Profit) ---
        self.STOP_LOSS_FACTOR = 3.0
        self.TAKE_PROFIT_FACTOR = 1.0
        
        # Action: Continuous value from -1.0 (Max Short) to +1.0 (Max Long)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space (16 features: Signals, Regime, State, Financials)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )
        
        # REWARD PARAMETERS (Sharpe-Optimized)
        self.reward_scale = 10.0
        self.drawdown_penalty_factor = 1.0 
        self.holding_penalty_factor = 0.005 
        self.profit_bonus = 5.0 
        self.VOLATILITY_WINDOW = 60 # Window size for volatility calculation (No longer directly used for reward)
        
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
        """Compute spread and advanced features including ADF and Half-Life"""
        x = self.data.iloc[:, 0]
        y = self.data.iloc[:, 1]
        
        self.spread = x - y
        
        # 1. Mean Reversion & Volatility Signals
        rolling_mean = self.spread.rolling(self.lookback).mean()
        rolling_std = self.spread.rolling(self.lookback).std() + 1e-8
        
        self.zscore_short = (self.spread - rolling_mean) / rolling_std
        self.zscore_long = ((self.spread - self.spread.rolling(self.lookback * 2).mean()) / 
                             (self.spread.rolling(self.lookback * 2).std() + 1e-8))
        self.vol_short = rolling_std
        self.vol_long = self.spread.rolling(self.lookback * 3).std()
        self.vol_ratio = self.vol_short / (self.vol_long + 1e-8)
        self.rsi = self._compute_rsi(self.spread, period=14)

        # 4. Stationarity/Regime Features 
        adf_pvalue = self.spread.rolling(self.lookback).apply(lambda s: adfuller(s.dropna())[1], raw=False)
        half_life_series = self.spread.rolling(self.lookback).apply(lambda s: half_life(s.dropna()), raw=False)

        self.adf_pvalue_np = np.nan_to_num(adf_pvalue.to_numpy(), nan=1.0)
        self.half_life_np = np.nan_to_num(half_life_series.to_numpy(), nan=252.0)

        # Convert to numpy and fill NaNs
        self.spread_np = np.nan_to_num(self.spread.to_numpy(), nan=0.0)
        self.zscore_short_np = np.nan_to_num(self.zscore_short.to_numpy(), nan=0.0)
        self.zscore_long_np = np.nan_to_num(self.zscore_long.to_numpy(), nan=0.0)
        self.vol_np = np.nan_to_num(self.vol_short.to_numpy(), nan=1.0)
        self.vol_ratio_np = np.nan_to_num(self.vol_ratio.to_numpy(), nan=1.0)
        self.rsi_np = np.nan_to_num(self.rsi.to_numpy(), nan=50.0)
        
    def _get_observation(self, idx: int) -> np.ndarray:
        """Build NORMALIZED observation vector"""
        if idx < 0 or idx >= len(self.spread_np):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        norm_unrealized = self.unrealized_pnl / self.initial_capital
        norm_realized = self.realized_pnl / self.initial_capital
        prev_idx = max(0, idx - 1)
        
        # Position is normalized by MAX_POSITION_UNITS for stability
        obs = np.array([
            self.zscore_short_np[idx],
            self.zscore_long_np[idx],
            self.vol_np[idx],
            self.spread_np[idx],
            self.rsi_np[idx] / 100.0,
            self.vol_ratio_np[idx],
            self.adf_pvalue_np[idx],
            self.half_life_np[idx] / 252.0,
            
            # --- AGENT STATE & NORMALIZED FINANCIALS ---
            float(self.position / self.MAX_POSITION_UNITS), # Normalized current position
            float(self.entry_spread) if self.position != 0 else 0.0,
            
            float(norm_unrealized),
            float(norm_realized),
            
            float(self.cash / self.initial_capital - 1),
            float(self.portfolio_value / self.initial_capital - 1),
            
            # --- LAGGED STATE FEATURES ---
            float(self.prev_position / self.MAX_POSITION_UNITS),
            float(self.spread_np[prev_idx]),
        ], dtype=np.float32)
        
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # When not in test_mode, start after lookback to ensure features are computed.
        # In test_mode, start at 0, and rely on the calling function to warm up the state.
        self.idx = self.lookback if not self.test_mode else 0 
        self.position = 0
        self.prev_position = 0.0
        self.entry_spread = 0.0
        self.days_in_position = 0
        
        self.cash = self.initial_capital
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.portfolio_value = self.initial_capital
        
        self.peak_value = self.initial_capital
        self.num_trades = 0
        self.trade_history = []
        
        self.prev_portfolio_value = self.initial_capital
        self.daily_returns_history = [] # Store all returns for stable volatility
        
        return self._get_observation(self.idx), {}

    def step(self, action: np.ndarray):
        """
        Execute one trading step with continuous action sizing and hard risk management.
        The action is a float array (e.g., [0.5]) in [-1.0, 1.0].
        """
        current_idx = self.idx
        self.prev_position = self.position
        
        is_last_step = (current_idx >= len(self.spread_np) - 1)
        current_spread = float(self.spread_np[current_idx])
        current_vol = float(self.vol_np[current_idx])
        
        if is_last_step:
            next_spread, next_idx = current_spread, current_idx
            target_position_from_agent = 0
        else:
            next_idx = current_idx + 1
            next_spread = float(self.spread_np[next_idx])
            
            # --- CRITICAL: Continuous Action Interpretation ---
            action_value = np.clip(action[0], -1.0, 1.0)
            
            # 1. Desired position based on max units
            desired_position_units = round(action_value * self.MAX_POSITION_UNITS)
            
            # 2. Notional Constraint (Risk Management)
            max_notional_value = self.initial_capital * self.MAX_POSITION_CAP
            estimated_notional = abs(desired_position_units * current_spread)
            
            if estimated_notional > max_notional_value:
                scaling_factor = max_notional_value / estimated_notional
                target_position_from_agent = round(desired_position_units * scaling_factor)
            else:
                target_position_from_agent = desired_position_units

        realized_pnl_this_step = 0.0
        transaction_costs = 0.0
        
        # --- Hard Risk Management Check (SL/TP) ---
        forced_exit = False
        if self.position != 0 and self.entry_spread != 0.0:
            pnl_on_current_spread = self.position * (current_spread - self.entry_spread)
            
            sl_threshold = abs(self.position) * current_vol * self.STOP_LOSS_FACTOR
            if pnl_on_current_spread < -sl_threshold:
                target_position_from_agent = 0
                forced_exit = True
            
            tp_threshold = abs(self.position) * current_vol * self.TAKE_PROFIT_FACTOR
            if pnl_on_current_spread > tp_threshold:
                target_position_from_agent = 0
                forced_exit = True

        target_position = target_position_from_agent

        # 4. Execute Trade & Update Financials
        position_change = target_position - self.position
        trade_occurred = (position_change != 0)
        
        if trade_occurred:
            if self.position != 0:
                spread_change = current_spread - self.entry_spread
                
                # Closed size handles partial closure/reversal correctly
                closed_size = abs(self.position)
                if abs(target_position) < abs(self.position) and np.sign(target_position) == np.sign(self.position):
                    closed_size = abs(position_change)

                realized_pnl_this_step = (self.position / abs(self.position)) * closed_size * spread_change
                
            trade_size = abs(position_change)
            current_price_x = float(self.data.iloc[current_idx, 0])
            current_price_y = float(self.data.iloc[current_idx, 1])
            notional = trade_size * (current_price_x + current_price_y) / 2
            
            transaction_costs = notional * self.transaction_cost_rate
            self.num_trades += 1
            
            # Reset Entry Price/Days logic
            if target_position != 0 and (self.position == 0 or np.sign(target_position) != np.sign(self.position)):
                self.entry_spread = current_spread
                self.days_in_position = 0
            elif target_position == 0:
                self.entry_spread = 0.0
                self.days_in_position = 0
                
            if self.position != 0 and target_position == 0:
                self.trade_history.append({
                    'entry_spread': self.entry_spread, 'exit_spread': current_spread,
                    'position': self.position, 'pnl': realized_pnl_this_step,
                    'holding_days': self.days_in_position, 'forced_close': is_last_step or forced_exit
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
        daily_return = (self.portfolio_value - self.prev_portfolio_value) / max(self.prev_portfolio_value, 1e-8)
        self.prev_portfolio_value = self.portfolio_value
        self.daily_returns_history.append(daily_return) # Track daily return
        
        # 6. Metrics
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-8)
        
        # 7. Reward (Simplified: Scaled Daily Return + Drawdown Penalty) ⬅️ MODIFIED

        # Scale daily return (e.g., multiply by 1000 to get a meaningful reward magnitude)
        daily_return_scaled = daily_return * 1000.0 

        # Increased penalty scale for faster learning
        drawdown_penalty = self.drawdown_penalty_factor * drawdown * 5.0 
        holding_penalty = self.holding_penalty_factor * self.days_in_position * (1.0 if self.position != 0 else 0.0)

        profit_bonus = 0.0
        if realized_pnl_this_step > 0:
            # Bonus for realized profit, scaled by initial capital
            profit_bonus = self.profit_bonus * (realized_pnl_this_step / self.initial_capital)

        # The Reward Formula: Raw Return - Drawdown Penalty - Holding Cost + Realized Bonus
        reward = daily_return_scaled - drawdown_penalty - holding_penalty + profit_bonus
        
        reward = np.clip(reward, -self.reward_scale, self.reward_scale)
        
        # 8. Index and 9. Obs
        if not is_last_step: self.idx = next_idx
        obs = self._get_observation(self.idx)
        
        # 10. Info
        info = {
            'portfolio_value': float(self.portfolio_value), 'cash': float(self.cash),
            'realized_pnl': float(self.realized_pnl), 'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl_this_step': float(realized_pnl_this_step), 'transaction_costs': float(transaction_costs),
            'position': int(self.position), 'entry_spread': float(self.entry_spread),
            'current_spread': float(current_spread), 'days_in_position': int(self.days_in_position),
            'daily_return': float(daily_return), 'drawdown': float(drawdown),
            'num_trades': int(self.num_trades), 'trade_occurred': bool(trade_occurred),
            'cum_return': float(self.portfolio_value / self.initial_capital - 1),
            'forced_close': is_last_step and trade_occurred,
            'risk_exit': bool(forced_exit),
            'adf_pvalue': float(self.adf_pvalue_np[current_idx]),
            'half_life': float(self.half_life_np[current_idx]), 
            'price_x': float(self.data.iloc[current_idx, 0]), 'price_y': float(self.data.iloc[current_idx, 1])
        }
        
        terminated = is_last_step
        return obs, float(reward), terminated, False, info

# ===================================================================
# 3. OPERATOR AGENT (PPO-LSTM Wrapper)
# ===================================================================

@dataclass
class OperatorAgent:
    
    logger: JSONLogger = None
    storage_dir: str = "models/"
    save_detailed_trace = save_detailed_trace 

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        self.active = True
        self.transaction_cost = CONFIG.get("transaction_cost", 0.0005)
        self.current_step = 0
        self.traces_buffer = []
        self.max_buffer_size = 1000

    def get_current_step(self): return self.current_step
    def get_traces_since_step(self, start_step): return [t for t in self.traces_buffer if t.get('step', 0) >= start_step]
    def add_trace(self, trace):
        self.traces_buffer.append(trace)
        if len(self.traces_buffer) > self.max_buffer_size: self.traces_buffer = self.traces_buffer[-self.max_buffer_size:]
    def clear_traces_before_step(self, step): self.traces_buffer = [t for t in self.traces_buffer if t.get('step', 0) >= step]
    
    def apply_command(self, command):
        cmd_type = command.get("command")
        if cmd_type == "pause":
            self.active = False
            if self.logger: self.logger.log("operator", "paused", {})
        elif cmd_type == "resume":
            self.active = True
            if self.logger: self.logger.log("operator", "resumed", {})

    def load_model(self, model_path): return RecurrentPPO.load(model_path)

    def train_on_pair(self, prices: pd.DataFrame, x: str, y: str,
                      lookback: int = None, timesteps: int = None,
                      shock_prob: float = None, shock_scale: float = None,
                      use_curriculum: bool = False):

        seed = CONFIG.get("random_seed", 42)
                                 
        if not self.active: return None

        if lookback is None: lookback = CONFIG.get("rl_lookback", 30)
        if timesteps is None: timesteps = CONFIG.get("rl_timesteps", 500000)

        series_x = prices[x]
        series_y = prices[y]

        print(f"\n{'='*70}")
        print(f"Training pair: {x} - {y} (LSTM POLICY)")
        print(f"  Timesteps (RL): {timesteps:,}")
        print(f"{'='*70}")

        # Note: position_scale is removed for the continuous environment (V3)
        env = PairTradingEnv(
            series_x, series_y, lookback, initial_capital=10000,
            transaction_cost_rate=0.0005, test_mode=False
        )
        env.reset(seed=seed)

        policy_kwargs = dict(lstm_hidden_size=512, n_lstm_layers=1)

        model = RecurrentPPO(
            "MlpLstmPolicy", env,
            learning_rate=0.0001, n_steps=4096, batch_size=256, n_epochs=10, gamma=0.99,
            ent_coef=0.02, clip_range=0.1, verbose=1, device="auto", seed=seed, policy_kwargs=policy_kwargs
        )

        model.learn(total_timesteps=timesteps)

        model_path = os.path.join(self.storage_dir, f"operator_model_{x}_{y}.zip")
        model.save(model_path)
        print(f"\n✅ Model saved to {model_path}")

        # Evaluation
        env_eval = PairTradingEnv(series_x, series_y, lookback, initial_capital=10000,
                                     transaction_cost_rate=0.0005, test_mode=False)
        
        obs, _ = env_eval.reset()
        done = False
        daily_returns, positions = [], []
        lstm_states, episode_starts = None, np.ones((1,), dtype=bool)

        while not done:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, reward, done, _, info = env_eval.step(action)
            episode_starts = np.array([done])
            daily_returns.append(info.get('daily_return', 0))
            positions.append(info.get('position', 0))

        sharpe = calculate_sharpe(daily_returns)
        sortino = calculate_sortino(daily_returns)
        final_return = (env_eval.portfolio_value / env_eval.initial_capital - 1) * 100

        print(f"\n📈 Training Results: Return={final_return:.2f}%, Sharpe={sharpe:.3f}, Sortino={sortino:.3f}")

        trace = {
            "pair": (x, y), "cum_return": final_return, 
            "max_drawdown": (env_eval.peak_value - env_eval.portfolio_value) / env_eval.peak_value,
            "sharpe": sharpe, "sortino": sortino, "model_path": model_path,
            "positions_used": np.unique(positions).tolist()
        }

        if self.logger: self.logger.log("operator", "pair_trained", trace)
        return trace

# ===================================================================
# 4. WORKFLOW FUNCTIONS (TRAIN/HOLDOUT)
# ===================================================================

def train_operator_on_pairs(operator: OperatorAgent, prices: pd.DataFrame,
                            pairs: list, max_workers: int = None):

    if max_workers is None: max_workers = CONFIG.get("max_workers", 2)
    all_traces = []

    def train(pair):
        x, y = pair
        return operator.train_on_pair(prices, x, y)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train, pair) for pair in pairs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Operator Training"):
            result = f.result()
            if result: all_traces.append(result)

    save_path = os.path.join(operator.storage_dir, "all_operator_traces.json")
    with open(save_path, "w") as f: json.dump(all_traces, f, indent=2, default=str)

    if operator.logger: operator.logger.log("operator", "batch_training_complete", {"n_pairs": len(all_traces)})
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for trace in all_traces:
        print(f"{trace['pair'][0]}-{trace['pair'][1]}: Return={trace['cum_return']:.2f}%, Sharpe={trace['sharpe']:.2f}")
    print("="*70)
    return all_traces


def run_operator_holdout(operator, holdout_prices, pairs, supervisor):
    """
    Run holdout testing with supervisor monitoring, including a warm-up loop
    to correctly initialize features and the RecurrentPPO LSTM state.
    """
    if "supervisor_rules" in CONFIG and "holdout" in CONFIG["supervisor_rules"]:
        check_interval = CONFIG["supervisor_rules"]["holdout"].get("check_interval", 20)
    else:
        check_interval = 20
    
    operator.traces_buffer = []
    operator.current_step = 0

    global_step = 0
    all_traces = []
    skipped_pairs = []

    lookback = CONFIG.get("rl_lookback", 30)
    
    # The required warm-up steps align with the history added in the main script (90 days).
    WARM_UP_STEPS = 90 

    for pair in pairs:
        print(f"\n{'='*70}")
        print(f"Testing pair: {pair[0]} - {pair[1]}")
        print(f"{'='*70}")

        if pair[0] not in holdout_prices.columns or pair[1] not in holdout_prices.columns:
            print(f"⚠️ Warning: Tickers {pair} not found in holdout data - skipping")
            skipped_pairs.append({"pair": f"{pair[0]}-{pair[1]}", "reason": "Data not found", "severity": "skip"})
            continue

        series_x = holdout_prices[pair[0]].dropna()
        series_y = holdout_prices[pair[1]].dropna()
        aligned = pd.concat([series_x, series_y], axis=1).dropna()

        if len(aligned) < WARM_UP_STEPS + 1:
            print(f"⚠️ Insufficient data ({len(aligned)} steps total, requires at least {WARM_UP_STEPS + 1}) - skipping")
            skipped_pairs.append({"pair": f"{pair[0]}-{pair[1]}", "reason": "Insufficient data", "severity": "skip"})
            continue

        model_path = os.path.join(operator.storage_dir, f"operator_model_{pair[0]}_{pair[1]}.zip")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found - skipping")
            skipped_pairs.append({"pair": f"{pair[0]}-{pair[1]}", "reason": "Model not found", "severity": "skip"})
            continue

        model = operator.load_model(model_path)
        print(f"  ✓ Model loaded")

        # TEST ENVIRONMENT (V3)
        env = PairTradingEnv(
              series_x=aligned.iloc[:, 0], series_y=aligned.iloc[:, 1], 
              lookback=lookback, initial_capital=10000,
              transaction_cost_rate = 0.0005, test_mode=True
            )

        episode_traces = []
        local_step = 0
        obs, info = env.reset() # env.idx = 0 here

        # --- WARM-UP LOOP: Advance the environment index and LSTM state ---
        print(f"  ✓ Warming up model state on {WARM_UP_STEPS} steps of history...")
        lstm_states, episode_starts = None, np.ones((1,), dtype=bool)
        
        # Run WARM_UP_STEPS through the environment and model
        for i in range(WARM_UP_STEPS):
            if env.idx >= len(env.spread_np) - 1:
                print("  ⚠️ Data ended during warm-up. Skipping pair.")
                break 
                
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            # Advance environment state (features, index)
            obs, _, _, _, _ = env.step(action) 
            episode_starts = np.array([False]) 
        
        # --- CRITICAL CLEAN RESET ---
        # 1. Store the index pointing to the first day of actual trading.
        start_idx_for_trading = env.idx 
        
        # 2. Reset financial state (Cash, PnL, Position) to initial values.
        # This preserves the features and the LSTM state.
        env.reset() 
        
        # 3. Restore the index to the position where warm-up ended.
        env.idx = start_idx_for_trading 
        print(f"  ✓ Warm-up complete. Financial state reset. Starting live trading loop from index {env.idx}.")
        # --- END CRITICAL CLEAN RESET ---


        terminated = False
        skip_to_next_pair = False
        
        # Trading loop with supervisor monitoring
        while not terminated and not skip_to_next_pair:
            # Predict uses the LSTM state carried over from the warm-up
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            
            obs, reward, terminated, _, info = env.step(action)
            episode_starts = np.array([terminated])

            # --- Trace logging starts here (after warm-up) ---
            trace = {
                "pair": f"{pair[0]}-{pair[1]}", "step": global_step, "local_step": local_step, 
                "reward": float(reward), "portfolio_value": float(info.get("portfolio_value", 0.0)),
                "cum_return": float(info.get("cum_return", 0.0)), "position": float(info.get("position", 0)),
                "max_drawdown": float(info.get("drawdown", 0)), "cash": float(info.get("cash", 0.0)),
                "realized_pnl": float(info.get("realized_pnl", 0.0)), "unrealized_pnl": float(info.get("unrealized_pnl", 0.0)),
                "realized_pnl_this_step": float(info.get("realized_pnl_this_step", 0.0)), 
                "transaction_costs": float(info.get("transaction_costs", 0.0)),
                "entry_spread": float(info.get("entry_spread", 0.0)), "current_spread": float(info.get("current_spread", 0.0)),
                "days_in_position": int(info.get("days_in_position", 0)), "daily_return": float(info.get("daily_return", 0.0)),
                "num_trades": int(info.get("num_trades", 0)), "trade_occurred": bool(info.get("trade_occurred", False)),
                "risk_exit": bool(info.get("risk_exit", False)),
                "price_x": float(info.get("price_x", 0.0)), "price_y": float(info.get("price_y", 0.0))
            }

            episode_traces.append(trace)
            all_traces.append(trace)
            operator.add_trace(trace)
            
            if hasattr(operator, 'save_detailed_trace'): operator.save_detailed_trace(trace)
            if operator.logger: operator.logger.log("operator", "holdout_step", trace)
            # --- End Trace Logging ---

            # SUPERVISOR MONITORING (every N steps)
            if local_step > 0 and local_step % check_interval == 0:
                decision = supervisor.check_operator_performance(episode_traces, pair, phase="holdout")
                
                if decision["action"] == "stop":
                    severity = decision.get("severity", "critical")
                    print(f"\n⛔ SUPERVISOR INTERVENTION [{severity.upper()}]: Skipping to next pair")
                    print(f"    Reason: {decision['reason']}")
                    print(f"    Metrics: {decision['metrics']}")
                    
                    skip_info = {
                        "pair": f"{pair[0]}-{pair[1]}", "reason": decision['reason'],
                        "severity": severity, "step_stopped": global_step,
                        "local_step_stopped": local_step, "metrics": decision['metrics']
                    }
                    skipped_pairs.append(skip_info)
                    skip_to_next_pair = True
                    if operator.logger: operator.logger.log("supervisor", "intervention", skip_info)
                    continue
                
                elif decision["action"] == "adjust":
                    print(f"\n⚠️  SUPERVISOR WARNING [{decision.get('severity', 'warning').upper()}]:")
                    print(f"    {decision['reason']}")
                    if 'suggestion' in decision: print(f"    💡 Suggestion: {decision['suggestion']}")
                
                elif decision["action"] == "warn":
                    if local_step % (check_interval * 4) == 0:
                        print(f"\nℹ️  SUPERVISOR INFO:")
                        print(f"    {decision['reason']}")
                
                if local_step % (check_interval * 2) == 0:
                    metrics = decision["metrics"]
                    print(f"\n📊 Step {local_step}: DD={metrics.get('drawdown', 0):.2%}, Sharpe={metrics.get('sharpe', 0):.2f}")

            local_step += 1
            global_step += 1
            operator.current_step = global_step

        # END OF PAIR SUMMARY
        if skip_to_next_pair:
            print(f"⏭️  Pair skipped early at step {local_step}")
        else:
            print(f"  ✓ Complete: {len(episode_traces)} steps")
        
        # Sharpe and Sortino
        sharpe = calculate_sharpe(episode_traces)
        sortino = calculate_sortino(episode_traces)

        # Logging final metrics (Assuming the original logging logic is complex)
        if operator.logger and len(episode_traces) > 0:
            final_cum_return = episode_traces[-1].get('cum_return', 0)
            final_pnl = episode_traces[-1].get('realized_pnl', 0)
            operator.logger.log("operator", "episode_complete", {
                "pair": f"{pair[0]}-{pair[1]}", "total_steps": len(episode_traces),
                "final_cum_return": final_cum_return, "total_pnl": final_pnl,
                "sharpe": sharpe, "sortino": sortino, "was_skipped": skip_to_next_pair
            })
            
    print("\n" + "="*70)
    print("HOLDOUT TESTING COMPLETE")
    print("="*70)
    print(f"Total steps: {global_step}")
    return all_traces, skipped_pairs
