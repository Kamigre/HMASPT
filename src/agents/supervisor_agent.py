import os
import json
import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import google.generativeai as genai
from statsmodels.tsa.stattools import adfuller

from config import CONFIG
from agents.message_bus import JSONLogger
from utils import half_life as compute_half_life, compute_spread

@dataclass
class SupervisorAgent:
    
    logger: JSONLogger = None
    df: pd.DataFrame = None  # Full price data for validation
    storage_dir: str = "./storage"
    gemini_api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    use_gemini: bool = True

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize Gemini for explanations
        if self.use_gemini:
            try:
                api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    generation_config = {
                        "temperature": self.temperature,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 2048,
                    }
                    self.client = genai.GenerativeModel(
                        model_name=self.model,
                        generation_config=generation_config
                    )
                    print(f"✅ Gemini API initialized")
                else:
                    self.use_gemini = False
                    print("⚠️ No Gemini API key - using fallback explanations")
            except Exception as e:
                self.use_gemini = False
                print(f"⚠️ Gemini init failed: {e}")
        
        self._log("init", {
            "gemini_enabled": self.use_gemini,
            "supervisor_rules_loaded": "supervisor_rules" in CONFIG
        })

    def _log(self, event: str, details: Dict[str, Any]):
        """Simple logging wrapper."""
        if self.logger:
            self.logger.log("supervisor", event, details)

    def format_skip_info(self, pair: Tuple[str, str], decision: Dict, step: int) -> Dict:
        """Format skip information for visualization."""
        return {
            "pair": f"{pair[0]}-{pair[1]}",
            "reason": decision['reason'],
            "step_stopped": step,
            "metrics": decision['metrics']
        }

    # ===================================================================
    # 1. PAIR VALIDATION (used by Selector)
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

        # Pivot price data
        prices = self.df.pivot(
            index="date",
            columns="ticker",
            values="adj_close"
        ).sort_index()

        print(f"\n🔍 Validating {len(df_pairs)} pairs...")
        
        for idx, row in df_pairs.iterrows():
            x, y = row["x"], row["y"]

            # Check tickers exist
            if x not in prices.columns or y not in prices.columns:
                continue

            # Extract validation window
            series_x = prices[x].loc[start:end].dropna()
            series_y = prices[y].loc[start:end].dropna()

            if min(len(series_x), len(series_y)) < 60:
                continue

            # Compute spread
            spread = compute_spread(series_x, series_y)
            if spread is None or len(spread) == 0:
                continue

            # --- Crossing frequency ---
            centered = spread - spread.mean()
            crossings = (centered.shift(1) * centered < 0).sum()
            days = (series_x.index[-1] - series_x.index[0]).days
            crossings_per_year = float(crossings) / max(days / 365.0, 1e-9)

            if crossings_per_year < min_crossings_per_year:
                continue

            # --- ADF test (stationarity) ---
            try:
                adf_res = adfuller(spread.dropna())
                adf_p = adf_res[1]
            except:
                adf_p = 1.0

            # --- Half-life (mean reversion speed) ---
            hl = compute_half_life(spread.values)
            try:
                hl_val = float(hl)
            except:
                hl_val = float("inf")

            # --- Pass criteria ---
            pass_criteria = (adf_p < 0.05) and (hl_val < half_life_max)

            validated.append({
                "x": x,
                "y": y,
                "score": float(row.get("score", np.nan)),
                "adf_p": float(adf_p),
                "half_life": hl_val,
                "crossings_per_year": float(crossings_per_year),
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
    # 2. OPERATOR MONITORING (Enhanced with Tiered Rules)
    # ===================================================================
    
    def check_operator_performance(
        self, 
        operator_traces: List[Dict[str, Any]],
        pair: Tuple[str, str],
        phase: str = "holdout"
    ) -> Dict[str, Any]:
        """
        Enhanced supervisor monitoring with tiered intervention.
        
        Args:
            operator_traces: List of trading step traces
            pair: Tuple of (ticker_x, ticker_y)
            phase: "training" or "holdout"
        
        Returns:
            Decision dict with:
                - action: "continue" | "warn" | "adjust" | "stop"
                - severity: "info" | "warning" | "critical"
                - reason: Human-readable explanation
                - suggestion: Optional improvement suggestion
                - metrics: Current performance metrics
        """
        
        # Get rules from CONFIG
        if "supervisor_rules" not in CONFIG:
            # Fallback to basic checks if rules not configured
            return self._basic_check(operator_traces, pair)
        
        rules = CONFIG["supervisor_rules"][phase]
        
        # Check minimum observations
        if len(operator_traces) < rules["min_observations"]:
            return {
                "action": "continue",
                "severity": "info",
                "reason": "insufficient_data",
                "metrics": {}
            }
        
        # Filter out zero returns/pnls
        filtered_traces = [t for t in operator_traces if t.get("daily_return", 0) != 0 and t.get("realized_pnl_this_step", 0) != 0]

        returns = [t["daily_return"] for t in filtered_traces]
        pnls = [t["realized_pnl_this_step"] for t in filtered_traces]
        portfolio_values = [t.get("portfolio_value", 0) for t in operator_traces]  # keep full portfolio history for drawdown

        # Current state
        portfolio_value = portfolio_values[-1] if portfolio_values else 0
        max_portfolio_value = max(portfolio_values) if portfolio_values else 1
        max_dd = (max_portfolio_value - portfolio_value) / max(max_portfolio_value, 1e-8)

        # Sharpe ratio
        rf_daily = CONFIG.get("risk_free_rate", 0.04) / 252
        sharpe = 0.0
        if returns:
            excess_returns = np.array(returns) - rf_daily
            if len(excess_returns) > 1 and np.std(excess_returns, ddof=1) > 1e-8:
                sharpe = (np.mean(excess_returns) / np.std(excess_returns, ddof=1)) * np.sqrt(252)

        # Win rate
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0

        # Total P&L
        total_pnl = sum(pnls)

        # Package metrics
        metrics = {
            'n_observations': len(filtered_traces),
            'drawdown': max_dd,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'portfolio_value': portfolio_value,
            'avg_return': np.mean(returns) if returns else 0
        }

        # ============================================================
        # TIER 3: CRITICAL INTERVENTION - STOP
        # ============================================================
        if phase == "holdout" and "stop_tier" in rules:
            stop_tier = rules["stop_tier"]
            
            ### ALL ACTIONS WERE STOP

            if max_dd > stop_tier["catastrophic_drawdown"]:
                self._log("intervention_triggered", {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "action": "wan",
                    "severity": "critical",
                    "reason": "catastrophic_drawdown",
                    "drawdown": max_dd,
                    "threshold": stop_tier["catastrophic_drawdown"]
                })
                return {
                    'action': 'wan',
                    'severity': 'critical',
                    'reason': f'Drawdown {max_dd:.1%} exceeds critical limit {stop_tier["catastrophic_drawdown"]:.1%}',
                    'metrics': metrics
                }
            
            if sharpe < stop_tier["disastrous_sharpe"]:
                self._log("intervention_triggered", {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "action": "wan",
                    "severity": "critical",
                    "reason": "disastrous_sharpe",
                    "sharpe": sharpe,
                    "threshold": stop_tier["disastrous_sharpe"]
                })
                return {
                    'action': 'warn',
                    'severity': 'critical',
                    'reason': f'Sharpe {sharpe:.2f} below disastrous threshold {stop_tier["disastrous_sharpe"]:.2f}',
                    'metrics': metrics
                }
            
            if win_rate < stop_tier["consistent_failure"]:
                self._log("intervention_triggered", {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "action": "warn",
                    "severity": "critical",
                    "reason": "consistent_failure",
                    "win_rate": win_rate,
                    "threshold": stop_tier["consistent_failure"]
                })
                return {
                    'action': 'warn',
                    'severity': 'critical',
                    'reason': f'Win rate {win_rate:.1%} indicates consistent failure (threshold: {stop_tier["consistent_failure"]:.1%})',
                    'metrics': metrics
                }
            
            if total_pnl < stop_tier["runaway_losses"]:
                self._log("intervention_triggered", {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "action": "warn",
                    "severity": "critical",
                    "reason": "runaway_losses",
                    "total_pnl": total_pnl,
                    "threshold": stop_tier["runaway_losses"]
                })
                return {
                    'action': 'warn',
                    'severity': 'critical',
                    'reason': f'Total P&L ${total_pnl:.0f} indicates runaway losses (threshold: ${stop_tier["runaway_losses"]:.0f})',
                    'metrics': metrics
                }
        
        # ============================================================
        # TIER 2: ADJUSTMENT SUGGESTIONS - ADJUST
        # ============================================================
        if phase == "holdout" and "adjustment_tier" in rules:
            adjust_tier = rules["adjustment_tier"]
            
            if max_dd > adjust_tier["significant_drawdown"]:
                self._log("intervention_triggered", {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "action": "adjust",
                    "severity": "warning",
                    "reason": "significant_drawdown",
                    "drawdown": max_dd,
                    "threshold": adjust_tier["significant_drawdown"]
                })
                return {
                    'action': 'adjust',
                    'severity': 'warning',
                    'reason': f'Drawdown {max_dd:.1%} significant',
                    'suggestion': adjust_tier["suggestions"]["drawdown"],
                    'metrics': metrics
                }
            
            if sharpe < adjust_tier["very_low_sharpe"]:
                self._log("intervention_triggered", {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "action": "adjust",
                    "severity": "warning",
                    "reason": "very_low_sharpe",
                    "sharpe": sharpe,
                    "threshold": adjust_tier["very_low_sharpe"]
                })
                return {
                    'action': 'adjust',
                    'severity': 'warning',
                    'reason': f'Sharpe {sharpe:.2f} very low',
                    'suggestion': adjust_tier["suggestions"]["sharpe"],
                    'metrics': metrics
                }
            
            if win_rate < adjust_tier["terrible_win_rate"]:
                self._log("intervention_triggered", {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "action": "adjust",
                    "severity": "warning",
                    "reason": "terrible_win_rate",
                    "win_rate": win_rate,
                    "threshold": adjust_tier["terrible_win_rate"]
                })
                return {
                    'action': 'adjust',
                    'severity': 'warning',
                    'reason': f'Win rate {win_rate:.1%} terrible',
                    'suggestion': adjust_tier["suggestions"]["win_rate"],
                    'metrics': metrics
                }
        
        # ============================================================
        # TIER 1: INFORMATIVE WARNINGS - WARN
        # ============================================================
        if phase == "holdout" and "info_tier" in rules:
            info_tier = rules["info_tier"]
            warnings = []
            
            if max_dd > info_tier["moderate_drawdown"]:
                warnings.append(f'Drawdown {max_dd:.1%} elevated')
            
            if sharpe < info_tier["low_sharpe"]:
                warnings.append(f'Sharpe {sharpe:.2f} below target')
            
            if win_rate < info_tier["poor_win_rate"]:
                warnings.append(f'Win rate {win_rate:.1%} suboptimal')
            
            if warnings:
                self._log("performance_warning", {
                    "pair": f"{pair[0]}-{pair[1]}",
                    "action": "warn",
                    "warnings": warnings,
                    "metrics": metrics
                })
                return {
                    'action': 'warn',
                    'severity': 'info',
                    'reason': '; '.join(warnings),
                    'metrics': metrics
                }
        
        # All good - continue trading
        return {
            'action': 'continue',
            'severity': 'info',
            'reason': 'Performance within acceptable ranges',
            'metrics': metrics
        }

    def _basic_check(self, operator_traces: List[Dict[str, Any]], 
                     pair: Tuple[str, str]) -> Dict[str, Any]:
        """
        Fallback basic checks if supervisor_rules not in CONFIG.
        More lenient than the tiered system.
        """
        if len(operator_traces) < 10:
            return {"action": "continue", "reason": "insufficient_data"}
        
        returns = [t.get("daily_return", 0) for t in operator_traces]
        portfolio_values = [t.get("portfolio_value", 0) for t in operator_traces]
        
        portfolio_value = portfolio_values[-1] if portfolio_values else 0
        max_portfolio_value = max(portfolio_values) if portfolio_values else 1
        max_dd = (max_portfolio_value - portfolio_value) / max(max_portfolio_value, 1e-8)
        
        # Very basic threshold (50% drawdown)
        if max_dd > 0.50:
            return {
                "action": "stop",
                "reason": f"Extreme drawdown {max_dd:.2%} (fallback check)",
                "metrics": {"drawdown": max_dd}
            }
        
        return {
            "action": "continue",
            "reason": "performance_acceptable",
            "metrics": {"drawdown": max_dd}
        }

    # ===================================================================
    # 3. FINAL EVALUATION (after all trading complete)
    # ===================================================================
        
    def evaluate_portfolio(
        self, 
        operator_traces: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate complete portfolio performance after all trading.
        """

        # Group by pair
        traces_by_pair = {}
        for t in operator_traces:
            pair = t.get("pair", "unknown")
            if pair not in traces_by_pair:
                traces_by_pair[pair] = []
            traces_by_pair[pair].append(t)

        # ============================================================
        # Recalculate true daily returns using pnl / prev_value
        # ============================================================
        all_returns = []
        all_pnls = []

        for i in range(1, len(operator_traces)):
            pnl = operator_traces[i].get("realized_pnl_this_step", 0)
            pv_prev = operator_traces[i-1].get("portfolio_value", None)

            if pv_prev is None or pv_prev == 0 or pnl == 0:
                continue
            
            true_return = pnl / pv_prev
            if true_return == 0:
                continue  # skip zero returns entirely
            
            all_returns.append(true_return)
            all_pnls.append(pnl)

        # If no valid step, fallback safely
        if not all_returns:
            all_returns = []
            all_pnls = []

        total_pnl = sum(all_pnls)

        initial_pv = operator_traces[0].get("portfolio_value", 0.0)
        final_pv = operator_traces[-1].get("portfolio_value", 0.0)

        portfolio_cum_return = (final_pv - initial_pv) / initial_pv if initial_pv > 0 else 0.0

        sharpe = self._calculate_sharpe(all_returns) if all_returns else 0.0
        sortino = self._calculate_sortino(all_returns) if all_returns else 0.0

        # Risk metrics
        max_dd = max([t.get("max_drawdown", 0) for t in operator_traces] + [0])
        var_95 = float(np.percentile(all_returns, 5)) if all_returns else 0
        cvar_95 = float(np.mean([r for r in all_returns if r <= var_95])) if all_returns and any(r <= var_95 for r in all_returns) else var_95

        # Win rate
        positive = sum(1 for r in all_returns if r > 0)
        win_rate = positive / len(all_returns) if all_returns else 0

        # ============================================================
        # Per-pair summaries with cumulative returns
        # ============================================================
        pair_summaries = []
        for pair, traces in traces_by_pair.items():
            
            pair_pnls = []
            pair_returns = []

            for i in range(1, len(traces)):
                pnl = traces[i].get("realized_pnl_this_step", 0)
                pv_prev = traces[i-1].get("portfolio_value", None)

                if pv_prev is None or pv_prev == 0:
                    continue

                true_return = pnl / pv_prev
                pair_pnls.append(pnl)
                pair_returns.append(true_return)

            if not pair_returns:
                pair_returns = [0.0]
                pair_pnls = [0.0]

            initial_pv = traces[0].get("portfolio_value", 0.0)
            final_pv = traces[-1].get("portfolio_value", 0.0)

            pair_return = (final_pv - initial_pv) / initial_pv if initial_pv > 0 else 0.0
            
            pair_sharpe = self._calculate_sharpe(pair_returns)
            pair_sortino = self._calculate_sortino(pair_returns)
            pair_max_dd = max([t.get("max_drawdown", 0) for t in traces] + [0])

            pair_summaries.append({
                "pair": pair,
                "total_pnl": float(sum(pair_pnls)),
                "cum_return": float(pair_return),
                "sharpe": float(pair_sharpe),
                "sortino": float(pair_sortino),
                "max_drawdown": float(pair_max_dd),
                "steps": len(traces)
            })

        metrics = {
            "total_pnl": float(total_pnl),
            "cum_return": float(portfolio_cum_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "win_rate": float(win_rate),
            "avg_return": float(np.mean(all_returns)) if all_returns else 0,
            "std_return": float(np.std(all_returns)) if all_returns else 0,
            "n_pairs": len(traces_by_pair),
            "total_steps": len(operator_traces),
            "pair_summaries": pair_summaries
        }

        # Additional metrics
        metrics["positive_returns"] = sum(1 for r in all_returns if r > 0)
        metrics["negative_returns"] = sum(1 for r in all_returns if r < 0)
        metrics["median_return"] = float(np.median(all_returns)) if all_returns else 0.0
        metrics["avg_steps_per_pair"] = (
            metrics["total_steps"] / max(metrics["n_pairs"], 1)
        )

        # Generate actions based on final performance
        actions = self._generate_portfolio_actions(metrics)
        
        # Generate explanation
        explanation = self._generate_explanation(metrics, actions)
        
        summary = {
            "metrics": metrics,
            "actions": actions,
            "explanation": explanation
        }
        
        self._log("portfolio_evaluated", summary)
        
        # Save report
        report_path = os.path.join(self.storage_dir, "supervisor_final_report.json")
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary

    def _generate_portfolio_actions(self, metrics: Dict) -> List[Dict]:
        """
        Generate portfolio-level action recommendations.
        Uses CONFIG rules if available, otherwise fallback thresholds.
        """
        actions = []
        
        # Get thresholds from CONFIG or use defaults
        if "supervisor_rules" in CONFIG and "portfolio" in CONFIG["supervisor_rules"]:
            portfolio_rules = CONFIG["supervisor_rules"]["portfolio"]
            max_dd_threshold = portfolio_rules.get("max_portfolio_drawdown", 0.30)
            min_sharpe_threshold = portfolio_rules.get("min_portfolio_sharpe", -0.3)
        else:
            max_dd_threshold = 0.30
            min_sharpe_threshold = -0.3
        
        max_dd = metrics.get("max_drawdown", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate", 0)
        cvar_95 = metrics.get("cvar_95", 0)
        
        if max_dd > max_dd_threshold:
            actions.append({
                "action": "reduce_risk",
                "reason": f"Portfolio drawdown {max_dd:.1%} exceeded limit {max_dd_threshold:.1%}",
                "severity": "high"
            })
        
        if sharpe < min_sharpe_threshold:
            actions.append({
                "action": "review_strategy",
                "reason": f"Portfolio Sharpe {sharpe:.2f} below minimum {min_sharpe_threshold:.2f}",
                "severity": "high"
            })
        
        if win_rate < 0.45:
            actions.append({
                "action": "improve_entry_exit",
                "reason": f"Win rate {win_rate:.1%} below target 45%",
                "severity": "medium"
            })
        
        if cvar_95 < -0.05:
            actions.append({
                "action": "reduce_tail_risk",
                "reason": "Excessive tail losses detected (CVaR < -5%)",
                "severity": "high"
            })
        
        return actions

    # ===================================================================
    # HELPER METHODS
    # ===================================================================
    
    def _calculate_sharpe(self, returns: List[float]) -> float:

      returns = [r for r in returns if r != 0]
      if len(returns) < 2:
          return 0.0
      
      rf_daily = CONFIG.get("risk_free_rate", 0.04) / 252
      excess = np.array(returns) - rf_daily
      
      mean_excess = np.mean(excess)
      std_excess = np.std(excess, ddof=1)
      
      if std_excess < 1e-8:
          return 0.0
      
      return (mean_excess / std_excess) * np.sqrt(252)

    def _calculate_sortino(self, returns: List[float]) -> float:
      
        returns = [r for r in returns if r != 0]
        if len(returns) < 2:
            return 0.0
        
        rf_daily = CONFIG.get("risk_free_rate", 0.04) / 252
        excess = np.array(returns) - rf_daily
        
        mean_excess = np.mean(excess)
        downside = excess[excess < 0]
        
        if len(downside) == 0:
            return 100.0 if mean_excess > 0 else 0.0
        
        downside_std = np.sqrt(np.mean(downside**2))
        
        if downside_std < 1e-8:
            return 100.0 if mean_excess > 0 else 0.0
        
        return (mean_excess / downside_std) * np.sqrt(252)

    def _generate_explanation(self, metrics: Dict, actions: List[Dict]) -> str:
        """Generate natural language explanation using Gemini or fallback."""
        if not self.use_gemini:
            return self._fallback_explanation(metrics, actions)
        
        system_instruction = """You are an expert quantitative supervisor analyzing pairs trading performance. 
        Provide clear, actionable insights about risk and performance."""
        
        prompt = f"""Analyze this pairs trading portfolio and explain the results:
                    
                METRICS:
                {json.dumps(metrics, indent=2)}
                
                RECOMMENDED ACTIONS:
                {json.dumps(actions, indent=2)}
                
                Provide a concise 3-4 paragraph executive summary covering:
                1. Overall performance (Sharpe, Sortino, win rate, drawdown)
                2. Key risks or concerns
                3. Rationale for recommended actions
                4. Brief outlook
                
                Keep it professional and actionable. This is a simulated backtest for research purposes."""
        
        try:
            model = genai.GenerativeModel(
                model_name=self.model,
                generation_config={"temperature": self.temperature},
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            
            if response.candidates and response.candidates[0].content.parts:
                return "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))
            
        except Exception as e:
            print(f"⚠️ Gemini explanation failed: {e}")
        
        return self._fallback_explanation(metrics, actions)

    def _fallback_explanation(self, metrics: Dict, actions: List[Dict]) -> str:
        """Fallback explanation when Gemini unavailable."""
        text = "Portfolio Performance Summary:\n\n"
        
        text += f"The portfolio achieved a Sharpe ratio of {metrics['sharpe_ratio']:.2f} "
        text += f"and Sortino ratio of {metrics['sortino_ratio']:.2f} with a "
        text += f"win rate of {metrics['win_rate']:.1%}. "
        text += f"Maximum drawdown was {metrics['max_drawdown']:.2%}.\n\n"
        
        if actions:
            text += f"Risk Management: {len(actions)} intervention(s) recommended - "
            text += ", ".join([a['action'] for a in actions]) + ".\n\n"
        else:
            text += "No critical risk interventions required.\n\n"
        
        text += f"The portfolio traded {metrics['n_pairs']} pairs over "
        text += f"{metrics['total_steps']} steps with an average return of "
        text += f"{metrics['avg_return']:.4f} per step."
        
        return text
