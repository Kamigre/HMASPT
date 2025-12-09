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

    # ... (validate_pairs remains unchanged) ...

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
        if len(spread_history) > 20:
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

        # 2. Hard Drawdown Kill (> 20%)
        # Explicit kill switch regardless of strikes
        if metrics['drawdown'] > 0.20:
             return {
                'action': 'stop',
                'severity': 'critical',
                'reason': f'Hard Stop: Drawdown {metrics["drawdown"]:.1%} > 20%',
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
        if metrics['drawdown'] > 0.10: 
            violation = True
            violation_reason = f"Drawdown {metrics['drawdown']:.1%} > 10%"
        
        # Efficiency Threshold: Bad Sharpe after 15 days
        elif metrics['sharpe'] < -1.5 and days_in_pos > 15: 
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
            # Heal strikes if performance recovers (drawdown < 5%)
            if self.monitoring_state[pair_key]['strikes'] > 0 and metrics['drawdown'] < 0.05:
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

    # ... (evaluate_portfolio and helpers remain unchanged) ...
    def _calculate_sharpe(self, returns: List[float]) -> float:
        if len(returns) < 2: return 0.0
        rf = CONFIG.get("risk_free_rate", 0.04) / 252
        exc = np.array(returns) - rf
        std = np.std(exc, ddof=1)
        return (np.mean(exc) / std) * np.sqrt(252) if std > 1e-8 else 0.0
