import os
import json
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import google.generativeai as genai
from statsmodels.tsa.stattools import adfuller

# Local imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import CONFIG
    from agents.message_bus import JSONLogger
    from utils import half_life as compute_half_life, compute_spread
except ImportError:
    CONFIG = {"risk_free_rate": 0.04}
    JSONLogger = None
    compute_half_life = lambda x: 10
    compute_spread = lambda x, y: x - y

@dataclass
class SupervisorAgent:
    
    logger: Optional[JSONLogger] = None
    df: pd.DataFrame = None 
    storage_dir: str = "./storage"
    gemini_api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    use_gemini: bool = True
    
    # Check frequency: Every 3 days
    check_frequency: int = 3 
    
    monitoring_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        # Gemini setup
        if self.use_gemini:
            try:
                api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.client = genai.GenerativeModel(model_name=self.model)
                else:
                    self.use_gemini = False
            except Exception:
                self.use_gemini = False

    def _log(self, event: str, details: Dict[str, Any]):
        if self.logger:
            self.logger.log("supervisor", event, details)

    # ===================================================================
    # 1. PAIR VALIDATION (Pre-Trading Check)
    # ===================================================================
    
    def validate_pairs(
        self, 
        df_pairs: pd.DataFrame, 
        validation_window: Tuple[pd.Timestamp, pd.Timestamp],
        half_life_max: float = 60,
        min_crossings_per_year: int = 12
    ) -> pd.DataFrame:
        
        start, end = validation_window
        validated = []

        # Pivot price data for fast access
        if self.df is None or self.df.empty:
            print("⚠️ Supervisor has no data for validation.")
            return df_pairs

        prices = self.df.pivot(
            index="date",
            columns="ticker",
            values="adj_close"
        ).sort_index()

        print(f"\n🔍 Validating {len(df_pairs)} pairs...")
        
        for idx, row in df_pairs.iterrows():
            x, y = row["x"], row["y"]

            if x not in prices.columns or y not in prices.columns:
                continue

            series_x = prices[x].loc[start:end].dropna()
            series_y = prices[y].loc[start:end].dropna()

            if min(len(series_x), len(series_y)) < 60:
                continue

            spread = compute_spread(series_x, series_y)
            if spread is None or len(spread) == 0:
                continue

            # Check stationarity (ADF Test)
            try:
                adf_res = adfuller(spread.dropna())
                adf_p = adf_res[1]
            except:
                adf_p = 1.0

            # Check mean reversion speed (Half-Life)
            hl = compute_half_life(spread.values)
            
            # Check Crossing Frequency
            centered = spread - spread.mean()
            crossings = (centered.shift(1) * centered < 0).sum()
            days = (series_x.index[-1] - series_x.index[0]).days
            crossings_per_year = float(crossings) / max(days / 252.0, 1e-9)

            # Decision Logic
            pass_criteria = (adf_p < 0.05) and (float(hl) < half_life_max) and (crossings_per_year >= min_crossings_per_year)

            validated.append({
                "x": x, "y": y,
                "score": float(row.get("score", np.nan)),
                "adf_p": float(adf_p),
                "half_life": float(hl),
                "crossings_per_year": crossings_per_year,
                "pass": bool(pass_criteria)
            })

        result_df = pd.DataFrame(validated)
        n_passed = result_df["pass"].sum() if len(result_df) > 0 else 0
        
        self._log("pairs_validated", {
            "n_total": len(df_pairs),
            "n_validated": len(result_df),
            "n_passed": int(n_passed)
        })
        
        print(f"✅ Validation complete: {n_passed}/{len(result_df)} pairs passed")
        return result_df

    # ===================================================================
    # 2. OPERATOR MONITORING
    # ===================================================================
    
    def check_operator_performance(
        self, 
        operator_traces: List[Dict[str, Any]],
        pair: Tuple[str, str],
        phase: str = "holdout"
    ) -> Dict[str, Any]:
        
        # 1. Config & Data Check
        rules = CONFIG.get("supervisor_rules", {}).get(phase, {}) if "supervisor_rules" in CONFIG else {}
        min_obs = rules.get("min_observations", 10)
        
        if len(operator_traces) < min_obs:
            return {"action": "continue", "severity": "info", "reason": "insufficient_data", "metrics": {}}

        pair_key = f"{pair[0]}-{pair[1]}"
        latest_trace = operator_traces[-1]
        days_in_pos = latest_trace.get('days_in_position', 0)
        
        # 2. State Initialization & Grace Period
        if pair_key not in self.monitoring_state:
            self.monitoring_state[pair_key] = {'strikes': 0, 'grace_period': True}

        # 3-day burn-in grace period
        if days_in_pos <= 3:
            self.monitoring_state[pair_key]['strikes'] = 0
            self.monitoring_state[pair_key]['grace_period'] = True
        else:
            self.monitoring_state[pair_key]['grace_period'] = False

        metrics = self._compute_live_metrics(operator_traces)
        
        # ============================================================
        # A. IMMEDIATE KILL (Structural Breaks) - CHECK EVERY DAY
        # ============================================================
        
        # 1. Structural Break (Z-Score > 3.0)
        spread_history = [t['current_spread'] for t in operator_traces]
        if len(spread_history) > 10:
            spread_series = pd.Series(spread_history)
            rolling_mean = spread_series.rolling(window=20).mean().iloc[-1]
            rolling_std = spread_series.rolling(window=20).std().iloc[-1]
            
            if rolling_std > 1e-8:
                current_z = abs(latest_trace['current_spread'] - rolling_mean) / rolling_std
                
                if current_z > 3.0:
                    self._log("intervention_triggered", {"pair": pair, "reason": "structural_break_zscore", "z": current_z})
                    return {
                        'action': 'stop',
                        'severity': 'critical',
                        'reason': f'Structural Break: Z-Score {current_z:.2f} > 3.0',
                        'metrics': metrics
                    }

        # 2. Hard Drawdown Kill (> 15%)
        # Explicit kill switch regardless of strikes
        if metrics['drawdown'] > 0.15:
             return {
                 'action': 'stop',
                 'severity': 'critical',
                 'reason': f'Hard Stop: Drawdown {metrics["drawdown"]:.1%} > 15%',
                 'metrics': metrics
             }

        # ============================================================
        # B. PERIODIC REVIEW (Strikes System)
        # ============================================================
        
        is_check_day = (days_in_pos > 0) and (days_in_pos % self.check_frequency == 0)
        
        if not is_check_day:
            return {'action': 'continue', 'severity': 'info', 'reason': 'off_cycle', 'metrics': metrics}

        # Stalemate Check (30 days)
        if days_in_pos > 30:
            unrealized_pnl = latest_trace.get('unrealized_pnl', 0.0)
            if unrealized_pnl <= 0:
                 return {
                     'action': 'stop', 
                     'severity': 'warning',
                     'reason': f'Stalemate ({days_in_pos} days) & Negative PnL. Capital rotation.',
                     'metrics': metrics
                 }

        # VIOLATION LOGIC
        violation = False
        violation_reason = ""
        
        # Warning Threshold: 10% Drawdown
        if metrics['drawdown'] > 0.1: 
            violation = True
            violation_reason = f"Drawdown {metrics['drawdown']:.1%} > 10%"
        
        # Efficiency Threshold: Bad Sharpe after 15 days
        elif metrics['sharpe'] < 0 and days_in_pos > 15: 
            violation = True
            violation_reason = f"Sharpe {metrics['sharpe']:.2f} (Inefficient Risk)"

        # TWO-STRIKE SYSTEM
        if violation:
            if self.monitoring_state[pair_key]['grace_period']:
                return {'action': 'continue', 'severity': 'info', 'reason': 'Grace Period', 'metrics': metrics}
            
            self.monitoring_state[pair_key]['strikes'] += 1
            strikes = self.monitoring_state[pair_key]['strikes']
            
            if strikes == 1:
                # STRIKE 1: WARN ONLY (No resizing)
                return {
                    'action': 'warn',
                    'severity': 'warning',
                    'reason': f'Strike 1/2: {violation_reason}. Monitoring closely.',
                    'metrics': metrics
                }
            elif strikes >= 2:
                # STRIKE 2: STOP
                return {
                    'action': 'stop',
                    'severity': 'critical',
                    'reason': f'Strike 2/2: {violation_reason}. Validation Failed.',
                    'metrics': metrics
                }
        else:
            # Heal strikes if performance recovers (drawdown < 2.5% - half of warning)
            if self.monitoring_state[pair_key]['strikes'] > 0 and metrics['drawdown'] < 0.025:
                self.monitoring_state[pair_key]['strikes'] -= 1
                
        return {
            'action': 'continue',
            'severity': 'info',
            'reason': 'Performance nominal',
            'metrics': metrics
        }
        
    def _compute_live_metrics(self, traces):
        returns = [t.get("daily_return", 0) for t in traces]
        portfolio_values = [t.get("portfolio_value", 0) for t in traces]
        
        current_pv = portfolio_values[-1] if portfolio_values else 0
        peak_pv = max(portfolio_values) if portfolio_values else 1
        
        drawdown = (peak_pv - current_pv) / max(peak_pv, 1e-8)
        
        return {
            'drawdown': drawdown,
            'sharpe': self._calculate_sharpe(returns),
            'total_steps': len(traces)
        }

    # ===================================================================
    # 3. FINAL EVALUATION (Post-Trading Aggregation)
    # ===================================================================
    
    def evaluate_portfolio(self, operator_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate complete portfolio performance using Global Dollar Aggregation.
        
        UPDATED LOGIC:
        - Aggregates 'portfolio_value' across all pairs to determine true dollar PnL.
        - Calculates returns based on (Global Dollar PnL / Total Invested Capital).
        - Generates an Equity Curve anchored strictly at 100.0.
        - Populates individual pair summaries (INCLUDES CUM_RETURN).
        """
        
        # --- 1. DATA PREPARATION ---
        processed_data = []
        for trace in operator_traces:
            # Filter for trade/step events, ignoring training events with list-type pairs
            if isinstance(trace.get('details', {}).get('pair'), list):
                continue
                
            base = trace.copy()
            details = base.pop('details', {})
            combined = {**base, **details}  
            processed_data.append(combined)

        df_all = pd.DataFrame(processed_data)
        
        if df_all.empty:
            return {
                "metrics": {
                    "total_pnl": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, 
                    "equity_curve": [], "pair_summaries": [],
                    "var_95": 0.0, "cvar_95": 0.0 # Added default values
                },
                "actions": [],
                "explanation": "No data available."
            }
            
        # Time alignment
        time_col = 'local_step' if 'local_step' in df_all.columns else 'timestamp'
        if time_col not in df_all.columns and 'step' in df_all.columns:
            time_col = 'step'
            
        # Ensure timestamp format if dealing with dates (consistency with visualizer)
        if time_col == 'timestamp':
            df_all[time_col] = pd.to_datetime(df_all[time_col])

        df_all = df_all.sort_values(by=time_col)
        
        # Ensure 'portfolio_value' exists (crucial for this logic)
        if 'portfolio_value' not in df_all.columns:
            # Fallback if trace doesn't have portfolio_value: estimate via cumulative PnL
            df_all['portfolio_value'] = df_all.get('realized_pnl', 0.0).cumsum() + 1000 # Dummy initial capital fallback

        # --- 2. GLOBAL DOLLAR AGGREGATION (MATCHING VISUALIZER) ---
        
        # Pivot to get Portfolio Value for all pairs at each step
        equity_matrix = df_all.pivot_table(index=time_col, columns='pair', values='portfolio_value')
        
        # Forward fill to handle asynchronous updates
        equity_matrix_ffill = equity_matrix.ffill()
        
        # Dollar PnL Change (Global)
        dollar_pnl_matrix = equity_matrix_ffill.diff()
        global_dollar_pnl = dollar_pnl_matrix.sum(axis=1).fillna(0.0)
        
        # Invested Capital (Shifted to represent capital at START of period)
        total_capital_series = equity_matrix_ffill.fillna(0.0).sum(axis=1)
        prev_capital = total_capital_series.shift(1)
        
        # Global Returns Calculation
        global_returns_series = global_dollar_pnl / prev_capital
        global_returns_series = global_returns_series.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        # --- 3. CONSTRUCT EQUITY CURVE (ANCHORED AT 100) ---
        
        # Calculate Cumulative Return Series
        cum_ret_series = (1 + global_returns_series).cumprod()
        portfolio_equity_curve = 100 * cum_ret_series
        
        # Explicitly prepend the start point (100.0) to match visualizer plotting logic
        equity_values = [100.0] + portfolio_equity_curve.tolist()
        
        # Generate corresponding labels/dates
        idx_list = portfolio_equity_curve.index.astype(str).tolist()
        equity_dates = ["Start"] + idx_list

        # Extract returns list for Sharpe/Sortino
        portfolio_returns = global_returns_series.tolist()

        # --- 4. CALCULATE PAIR SUMMARIES (No change to this block) ---
        
        pair_summaries = []
        unique_pairs = df_all['pair'].unique()

        for pair in unique_pairs:
            pair_trace = df_all[df_all['pair'] == pair].sort_values(by=time_col)
            if pair_trace.empty:
                continue

            # Determine returns for this specific pair
            if 'daily_return' in pair_trace.columns:
                pair_rets = pair_trace['daily_return'].tolist()
            else:
                pair_rets = pair_trace['portfolio_value'].pct_change().fillna(0.0).tolist()
            
            # PnL (Last realized PnL value)
            pair_pnl = pair_trace['realized_pnl'].iloc[-1] if 'realized_pnl' in pair_trace.columns else 0.0
            
            # Drawdown (Pair specific)
            pair_curve = np.cumprod([1 + r for r in pair_rets])
            pair_peak = np.maximum.accumulate(pair_curve)
            pair_dd = (pair_curve - pair_peak) / pair_peak
            pair_max_dd = abs(np.min(pair_dd)) if len(pair_dd) > 0 else 0.0

            # Win Rate (Pair specific)
            pair_win_rate = 0.0
            if 'realized_pnl_this_step' in pair_trace.columns:
                 closed = pair_trace[pair_trace['realized_pnl_this_step'] != 0]
                 if len(closed) > 0:
                     pair_win_rate = (closed['realized_pnl_this_step'] > 0).sum() / len(closed)

            # Cumulative Return (Pair specific)
            initial_val = pair_trace['portfolio_value'].iloc[0]
            final_val = pair_trace['portfolio_value'].iloc[-1]
            pair_cum_return = (final_val - initial_val) / initial_val if initial_val > 0 else 0.0

            pair_summaries.append({
                "pair": pair,
                "total_pnl": float(pair_pnl),
                "cum_return": float(pair_cum_return),
                "sharpe": self._calculate_sharpe(pair_rets),
                "sortino": self._calculate_sortino(pair_rets),
                "max_drawdown": float(pair_max_dd),
                "win_rate": pair_win_rate,
                "steps": len(pair_trace)
            })

        # --- 5. CALCULATE GLOBAL METRICS ---
        
        # Max Drawdown (Calculated on the anchored curve)
        normalized_equity_curve = pd.Series(equity_values)
        running_max = normalized_equity_curve.cummax()
        dd_series = (normalized_equity_curve - running_max) / running_max
        portfolio_max_dd = abs(dd_series.min()) if not dd_series.empty else 0.0
        
        # Sharpe & Sortino (Based on the aggregated global returns)
        sharpe = self._calculate_sharpe(portfolio_returns)
        sortino = self._calculate_sortino(portfolio_returns)
        
        # VaR and CVaR Calculation (Matching Visualizer logic)
        if len(global_returns_series) > 0:
            # VaR 95% = 5th percentile of returns distribution
            var_95 = np.percentile(global_returns_series, 5)
            
            # CVaR 95% = Mean of returns falling below the VaR threshold
            cvar_returns = global_returns_series[global_returns_series <= var_95]
            cvar_95 = cvar_returns.mean() if len(cvar_returns) > 0 else var_95
        else:
            var_95 = 0.0
            cvar_95 = 0.0
        
        # Win Rate (Global Trades)
        win_rate = 0.0
        if 'realized_pnl_this_step' in df_all.columns:
            closed_trades = df_all[df_all['realized_pnl_this_step'] != 0]
            if not closed_trades.empty:
                wins = (closed_trades['realized_pnl_this_step'] > 0).sum()
                win_rate = wins / len(closed_trades)

        # Total PnL Calculation (Sum of Global Dollar PnL Series)
        total_pnl = float(global_dollar_pnl.sum())

        # Compile Metrics
        metrics = {
            "total_pnl": total_pnl,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": portfolio_max_dd,
            "avg_return": float(np.mean(portfolio_returns)) if portfolio_returns else 0.0,
            "cum_return": (normalized_equity_curve.iloc[-1] - 100) / 100,
            "win_rate": win_rate,
            "total_steps": len(df_all),
            "var_95": float(var_95),  # ADDED VAR
            "cvar_95": float(cvar_95), # ADDED CVAR
            "pair_summaries": pair_summaries
        }
        
        # Export curve data matching Visualizer
        metrics["equity_curve"] = equity_values
        metrics["equity_curve_dates"] = equity_dates

        # Actions are generated by the LLM in the explanation now.
        actions = [] 
        explanation = self._generate_explanation(metrics)
        
        # The LLM's explanation may contain suggested actions. We won't parse them back into the actions list, 
        # but keep the structure for compatibility.
        
        return {"metrics": metrics, "actions": actions, "explanation": explanation}

    # Removed the _generate_portfolio_actions method as requested.
    # def _generate_portfolio_actions(self, metrics: Dict) -> List[Dict]:
    #     ...

    def _calculate_sharpe(self, returns: List[float]) -> float:
        if len(returns) < 2: return 0.0
        rf = CONFIG.get("risk_free_rate", 0.04) / 252
        exc = np.array(returns) - rf
        std = np.std(exc, ddof=1)
        return (np.mean(exc) / std) * np.sqrt(252) if std > 1e-8 else 0.0

    def _calculate_sortino(self, returns: List[float]) -> float:
        if len(returns) < 2: return 0.0
        rf = CONFIG.get("risk_free_rate", 0.04) / 252
        exc = np.array(returns) - rf
        down = exc[exc < 0]
        std = np.sqrt(np.mean(down**2)) if len(down) > 0 else 0.0
        return (np.mean(exc) / std) * np.sqrt(252) if std > 1e-8 else 0.0

    def _generate_explanation(self, metrics: Dict) -> str:
        if not self.use_gemini:
            return self._fallback_explanation(metrics)
        
        # We need an estimate for 'avg_steps_per_pair' for the prompt's Risk Decomposition section.
        # Let's compute a simple average here before sending to the LLM.
        pair_steps = [p['steps'] for p in metrics.get('pair_summaries', [])]
        avg_steps_per_pair = np.mean(pair_steps) if pair_steps else 0

        prompt = f"""
                ### ROLE & OBJECTIVE
                Act as the Chief Risk Officer (CRO) of a Quantitative Hedge Fund. Your sole mandate is capital preservation and risk-adjusted growth. You are addressing the Investment Committee.
                
                ### INPUT DATA
                --- PORTFOLIO METRICS ---
                {json.dumps(metrics, indent=2, default=str)}
                
                --- ADDITIONAL CONTEXT ---
                Average Steps Per Pair: {avg_steps_per_pair:.2f}
                
                ### INSTRUCTIONS
                Produce a high-level, institutional-grade **Risk Memo**.
                * **Tone:** Clinical, academic, and extremely concise. No pleasantries, no "I hope this helps," and no email formatting (Subject/Dear Team).
                * **Format:** Bullet points and bold key figures only.
                * **Length:** Maximum 400 words.
                
                ### REQUIRED SECTIONS
                
                #### 1. Performance Attribution
                * **Return Efficiency:** Analyze Sharpe and Sortino ratios. Is the risk-adjusted return justifiable?
                * **Profit Quality:** Contrast Win Rate against Total PnL. Explicitly identify if the strategy suffers from "negative skew" (small frequent wins, rare massive losses).
                * **Distribution:** Evaluate the delta between Average Return and Median Return (Median is the 50th percentile of global_returns_series which is not explicitly provided, so use Avg. Return and Win Rate as proxies for skewness assessment).
                
                #### 2. Risk Decomposition
                * **Tail Risk Analysis:** Contrast Max Drawdown against VaR/CVaR (95%). Is the realized drawdown within modeled expectations?
                * **Duration/Stalemate Risk:** Analyze the Average Steps Per Pair. Is the holding period consistent with a mean-reversion thesis, or are capital costs eroding alpha?
                * **Concentration:** Identify if losses are systemic or isolated to specific pairs (Reference the pair_summaries).
                
                #### 3. CRO Verdict & Adjustments
                * **Action Mandate:** Based on the risk analysis, issue a mandate for the trading desk. This must be a list of 1-3 concrete, actionable steps (e.g., "Halt all trading," "Reduce capital allocation by 25%," "Liquidate pairs with Sharpe < 0").
                * **Traffic Light Signal:** Conclude with a single word: GREEN (Scale Up), YELLOW (Maintain/Monitor), or RED (De-risk/Halt).
                """
        
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception:
            return self._fallback_explanation(metrics)

    def _fallback_explanation(self, metrics: Dict) -> str:
        return f"Portfolio Sharpe: {metrics['sharpe_ratio']:.2f}. Drawdown: {metrics['max_drawdown']:.2%}. Win Rate: {metrics['win_rate']:.1%}."

    def _basic_check(self, operator_traces: List[Dict[str, Any]], pair: Tuple[str, str]) -> Dict[str, Any]:
        return {"action": "continue", "reason": "basic_check_pass", "metrics": {}}
