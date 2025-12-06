@dataclass
class SupervisorAgent:
    
    logger: JSONLogger = None
    df: pd.DataFrame = None
    storage_dir: str = "./storage"
    gemini_api_key: Optional[str] = None
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    use_gemini: bool = True

    def __post_init__(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        # ... [Keep Gemini init logic from before] ...
        
        # NEW: State tracking for sequential warnings
        # Format: { 'TickerA-TickerB': {'strikes': 0, 'last_check_step': 0} }
        self.monitoring_state = {} 

        self._log("init", {"gemini_enabled": self.use_gemini})

    def _log(self, event: str, details: Dict[str, Any]):
        if self.logger: self.logger.log("supervisor", event, details)

    # ... [Keep validate_pairs as is] ...

    def check_operator_performance(
        self, 
        operator_traces: List[Dict[str, Any]],
        pair: Tuple[str, str],
        phase: str = "holdout"
    ) -> Dict[str, Any]:
        
        pair_key = f"{pair[0]}-{pair[1]}"
        latest_trace = operator_traces[-1]
        
        # 1. Initialize State for this pair if new
        if pair_key not in self.monitoring_state:
            self.monitoring_state[pair_key] = {'strikes': 0, 'grace_period': True}

        # 2. Check for "Fresh" Trade (Reset strikes if position flipped or just opened)
        # If days_in_position is small, we consider it a new trade attempt
        days_in_pos = latest_trace.get('days_in_position', 0)
        if days_in_pos <= 5:
            # GRACE PERIOD: Reset strikes, be patient
            self.monitoring_state[pair_key]['strikes'] = 0
            self.monitoring_state[pair_key]['grace_period'] = True
        else:
            self.monitoring_state[pair_key]['grace_period'] = False

        # 3. Compute Metrics
        metrics = self._compute_live_metrics(operator_traces)
        
        # ============================================================
        # A. IMMEDIATE KILL (Structural Breaks) - No Mercy
        # ============================================================
        # If the Z-score is mathematically impossible (e.g. > 4.5), 
        # the model is broken. Stop immediately.
        spread_history = [t['current_spread'] for t in operator_traces]
        if len(spread_history) > 30:
            spread_series = pd.Series(spread_history)
            rolling_mean = spread_series.rolling(window=30).mean().iloc[-1]
            rolling_std = spread_series.rolling(window=30).std().iloc[-1]
            
            if rolling_std > 1e-8:
                current_z = abs(latest_trace['current_spread'] - rolling_mean) / rolling_std
                if current_z > 4.5:
                    return {
                        'action': 'stop',
                        'severity': 'critical',
                        'reason': f'Structural Break: Z-Score {current_z:.1f} > 4.5 (Instant Kill)',
                        'metrics': metrics
                    }

        # ============================================================
        # B. SEQUENTIAL WARNINGS (P&L Checks) - With Patience
        # ============================================================
        
        # Define Thresholds
        rules = CONFIG.get("supervisor_rules", {}).get(phase, {})
        stop_tier = rules.get("stop_tier", {})
        max_dd_limit = stop_tier.get("catastrophic_drawdown", 0.30)
        
        # Check for Violation
        violation = False
        violation_reason = ""
        
        if metrics['drawdown'] > max_dd_limit:
            violation = True
            violation_reason = f"Drawdown {metrics['drawdown']:.1%} > {max_dd_limit:.1%}"
        elif metrics['sharpe'] < -2.0 and days_in_pos > 20: # Only check Sharpe after 20 days
            violation = True
            violation_reason = f"Sharpe {metrics['sharpe']:.2f} is disastrous"

        # Apply Three-Strike Logic
        if violation:
            if self.monitoring_state[pair_key]['grace_period']:
                # IGNORING violation because we are in Grace Period
                return {
                    'action': 'continue',
                    'severity': 'info',
                    'reason': f'Grace Period: Ignoring {violation_reason}',
                    'metrics': metrics
                }
            
            # Increment Strikes
            self.monitoring_state[pair_key]['strikes'] += 1
            strikes = self.monitoring_state[pair_key]['strikes']
            
            if strikes == 1:
                return {
                    'action': 'warn',
                    'severity': 'warning',
                    'reason': f'Strike 1/3: {violation_reason}. Monitoring closely.',
                    'metrics': metrics
                }
            elif strikes == 2:
                return {
                    'action': 'adjust',
                    'severity': 'warning',
                    'reason': f'Strike 2/3: {violation_reason}. Persisting. Suggest size reduction.',
                    'suggestion': 'reduce_size_50_percent',
                    'metrics': metrics
                }
            elif strikes >= 3:
                return {
                    'action': 'stop',
                    'severity': 'critical',
                    'reason': f'Strike 3/3: {violation_reason}. Patience exhausted. Stopping.',
                    'metrics': metrics
                }
        else:
            # GOOD BEHAVIOR: Heal strikes slowly
            # If performance recovers, we forgive past sins
            if self.monitoring_state[pair_key]['strikes'] > 0:
                self.monitoring_state[pair_key]['strikes'] -= 1
                
        return {
            'action': 'continue',
            'severity': 'info',
            'reason': 'Performance nominal',
            'metrics': metrics
        }

    def _compute_live_metrics(self, traces):
        """Helper to calculate metrics efficiently."""
        returns = [t.get("daily_return", 0) for t in traces]
        portfolio_values = [t.get("portfolio_value", 0) for t in traces]
        
        current_pv = portfolio_values[-1] if portfolio_values else 0
        peak_pv = max(portfolio_values) if portfolio_values else 1
        drawdown = (peak_pv - current_pv) / max(peak_pv, 1e-8)
        
        return {
            'drawdown': drawdown,
            'sharpe': self._calculate_sharpe(returns),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
            'total_steps': len(traces)
        }

    # ... [Keep helper methods evaluate_portfolio, calculate_sharpe, etc.] ...
    def evaluate_portfolio(self, operator_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate complete portfolio performance."""
        
        traces_by_pair = {}
        for t in operator_traces:
            traces_by_pair.setdefault(t['pair'], []).append(t)

        all_returns = []
        all_pnls = []
        pair_summaries = []

        # Process each pair
        for pair, traces in traces_by_pair.items():
            pair_returns = []
            pair_pnls = []
            
            for i in range(1, len(traces)):
                pnl = traces[i].get("realized_pnl_this_step", 0)
                pv_prev = traces[i-1].get("portfolio_value", 0)
                
                if pv_prev > 0 and pnl != 0:
                    ret = pnl / pv_prev
                    pair_returns.append(ret)
                    all_returns.append(ret)
                    pair_pnls.append(pnl)
                    all_pnls.append(pnl)

            # Pair stats
            initial = traces[0]['portfolio_value']
            final = traces[-1]['portfolio_value']
            cum_ret = (final - initial) / initial if initial > 0 else 0
            
            pair_summaries.append({
                "pair": pair,
                "total_pnl": sum(pair_pnls),
                "cum_return": cum_ret,
                "sharpe": self._calculate_sharpe(pair_returns),
                "max_drawdown": max([t.get("max_drawdown", 0) for t in traces] + [0]),
                "steps": len(traces)
            })

        # Global stats
        metrics = {
            "total_pnl": sum(all_pnls),
            "sharpe_ratio": self._calculate_sharpe(all_returns),
            "sortino_ratio": self._calculate_sortino(all_returns),
            "max_drawdown": max([p['max_drawdown'] for p in pair_summaries] + [0]),
            "win_rate": sum(1 for r in all_returns if r > 0) / len(all_returns) if all_returns else 0,
            "avg_return": float(np.mean(all_returns)) if all_returns else 0,
            "total_steps": len(operator_traces),
            "n_pairs": len(traces_by_pair),
            "pair_summaries": pair_summaries
        }
        
        # Calculate Risk Metrics (VaR/CVaR)
        if all_returns:
            metrics["var_95"] = float(np.percentile(all_returns, 5))
            tail_losses = [r for r in all_returns if r <= metrics["var_95"]]
            metrics["cvar_95"] = float(np.mean(tail_losses)) if tail_losses else metrics["var_95"]
        else:
            metrics["var_95"] = 0.0
            metrics["cvar_95"] = 0.0

        # Additional activity metrics
        metrics["positive_returns"] = sum(1 for r in all_returns if r > 0)
        metrics["negative_returns"] = sum(1 for r in all_returns if r < 0)
        metrics["median_return"] = float(np.median(all_returns)) if all_returns else 0.0
        metrics["std_return"] = float(np.std(all_returns)) if all_returns else 0.0
        metrics["avg_steps_per_pair"] = metrics["total_steps"] / max(metrics["n_pairs"], 1)
        
        # Store cumulative return properly
        if operator_traces:
            start_pv = operator_traces[0].get("portfolio_value", 0)
            end_pv = operator_traces[-1].get("portfolio_value", 0)
            metrics["cum_return"] = (end_pv - start_pv) / start_pv if start_pv > 0 else 0
        else:
            metrics["cum_return"] = 0.0

        actions = self._generate_portfolio_actions(metrics)
        explanation = self._generate_explanation(metrics, actions)
        
        return {"metrics": metrics, "actions": actions, "explanation": explanation}

    def _generate_portfolio_actions(self, metrics: Dict) -> List[Dict]:
        actions = []
        if metrics['max_drawdown'] > 0.30:
            actions.append({"action": "reduce_risk", "reason": "Portfolio drawdown > 30%", "severity": "high"})
        if metrics['sharpe_ratio'] < 0:
            actions.append({"action": "halt_trading", "reason": "Negative Sharpe Ratio", "severity": "high"})
        return actions

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

    def _generate_explanation(self, metrics: Dict, actions: List[Dict]) -> str:
        if not self.use_gemini:
            return self._fallback_explanation(metrics, actions)
        
        prompt = f"""
        Act as a Quantitative Risk Manager. Analyze these pairs trading results:
        
        METRICS:
        {json.dumps(metrics, indent=2, default=str)}
        
        ACTIONS:
        {json.dumps(actions, indent=2)}
        
        Write a professional 3-paragraph executive summary:
        1. Performance Overview (Returns, Sharpe, Drawdown).
        2. Risk Analysis (Tail risk, worst pairs, structural breaks).
        3. Strategic Recommendation (Continue, Reduce Size, Halt).
        """
        
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception:
            return self._fallback_explanation(metrics, actions)

    def _fallback_explanation(self, metrics, actions):
        return f"Portfolio Sharpe: {metrics['sharpe_ratio']:.2f}. Drawdown: {metrics['max_drawdown']:.2%}. Win Rate: {metrics['win_rate']:.1%}."

    def _basic_check(self, operator_traces, pair):
        return {"action": "continue", "reason": "basic_check_pass", "metrics": {}}
