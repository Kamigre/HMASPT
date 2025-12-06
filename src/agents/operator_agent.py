import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Import CONFIG for risk-free rate
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import CONFIG

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

class PortfolioVisualizer:
    """Creates visual reports for pairs trading performance."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pairs"), exist_ok=True)
    
    def visualize_pair(self, traces: List[Dict], pair_name: str, was_skipped: bool = False, skip_info: Dict = None):
        """
        Create detailed visualization for a single pair, highlighting forced exits.
        """
        if len(traces) == 0:
            return
        
        # Extract data
        steps = [t['local_step'] for t in traces]
        pnls = [t['realized_pnl_this_step'] for t in traces]
        cum_return = [t['cum_return'] for t in traces]
        positions = [t['position'] for t in traces]
        drawdowns = [t['max_drawdown'] for t in traces]
        returns = [t['daily_return'] for t in traces]
        
        # Check for forced close in the last trace
        forced_close = traces[-1].get('forced_close', False)
        
        # Filter only trades (non-zero returns) for statistics
        trade_returns = [r for r in returns if r != 0]
        
        # Calculate metrics using only trades
        sharpe = self._calculate_sharpe(trade_returns) if trade_returns else 0.0
        sortino = self._calculate_sortino(trade_returns) if trade_returns else 0.0
        total_pnl = sum(pnls)
        final_return = cum_return[-1] * 100 if cum_return else 0
        max_dd = max(drawdowns) if drawdowns else 0
        
        # Get num_trades from last trace
        num_trades = traces[-1].get('num_trades', len(trade_returns))
        
        # Win rate calculation
        positive_returns = sum(1 for r in trade_returns if r > 0)
        win_rate = positive_returns / len(trade_returns) if trade_returns else 0
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Title with skip indicator
        title = f"Pair Trading Analysis: {pair_name}"
        if was_skipped:
            title += f" ⛔ STOPPED BY SUPERVISOR"
        fig.suptitle(title, fontsize=16, fontweight='bold', 
                     color='red' if was_skipped else 'black')
        
        # 1. Price Performance (was Cumulative Returns)
        ax1 = fig.add_subplot(gs[0, :])
        price_performance = 100 * (1 + np.array(cum_return))
        ax1.plot(steps, price_performance, linewidth=2, color='darkblue', label='Portfolio Value')
        ax1.axhline(100, color='black', linestyle='--', alpha=0.3)
        ax1.fill_between(steps, 100, price_performance, alpha=0.3, 
                         color='blue' if final_return > 0 else 'red')
        
        # MARKER: Forced Liquidation
        if forced_close:
             ax1.plot(steps[-1], price_performance[-1], 'rx', markersize=10, 
                      markeredgewidth=3, label="Forced Liquidation")
             ax1.legend(loc='upper left')
        
        ax1.set_title(f'Final Price Performance: {final_return:.2f}%', 
                      fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (Base 100)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.fill_between(steps, 0, np.array(drawdowns) * 100, alpha=0.5, color='red')
        ax2.plot(steps, np.array(drawdowns) * 100, linewidth=2, color='darkred')
        ax2.set_title(f'Drawdown (Max: {max_dd*100:.2f}%)', fontsize=11, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        # 3. Daily P&L
        ax3 = fig.add_subplot(gs[1, 1])
        colors = ['green' if p > 0 else 'red' for p in pnls]
        
        # Highlight forced close bar in Orange
        if forced_close:
            colors[-1] = 'orange'
            
        bar_width = max(1.0, len(steps) / 100) 
        ax3.bar(steps, pnls, color=colors, alpha=0.6, width=bar_width)
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title(f'Daily P&L (Total: ${total_pnl:.2f})', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Position Over Time
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(steps, positions, linewidth=2, color='purple', alpha=0.7)
        ax4.fill_between(steps, 0, positions, alpha=0.3, color='purple')
        ax4.axhline(0, color='black', linestyle='--', alpha=0.3)
        
        # ANNOTATION: Forced Exit
        if forced_close:
            ax4.annotate('Forced Exit', 
                         xy=(steps[-1], 0), 
                         xytext=(steps[-1], max(positions)*0.5 if max(positions)>0 else 50),
                         arrowprops=dict(facecolor='red', shrink=0.05),
                         fontsize=9, fontweight='bold', color='red', ha='right')

        ax4.set_title('Position Over Time', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Position Size')
        ax4.grid(True, alpha=0.3)
        
        # 5. Returns Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        if trade_returns:
            ax5.hist(trade_returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax5.set_title('Trade Returns Distribution', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Cumulative Sharpe
        ax6 = fig.add_subplot(gs[2, 1])
        if len(trade_returns) > 1:
            cumulative_sharpe = []
            for i in range(2, len(trade_returns) + 1):
                subset = trade_returns[:i]
                sharpe_i = self._calculate_sharpe(subset)
                cumulative_sharpe.append(sharpe_i)
            
            # Map back to steps roughly
            t_steps = range(2, len(trade_returns) + 1)
            ax6.plot(t_steps, cumulative_sharpe, linewidth=2, color='darkgreen')
            ax6.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax6.set_title(f'Cumulative Sharpe', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Cumulative Win Rate
        ax7 = fig.add_subplot(gs[2, 2])
        if trade_returns:
            cumulative_wr = []
            for i in range(1, len(trade_returns) + 1):
                subset = trade_returns[:i]
                wr_i = sum(1 for r in subset if r > 0) / len(subset)
                cumulative_wr.append(wr_i * 100)
            
            ax7.plot(range(1, len(trade_returns) + 1), cumulative_wr, linewidth=2, color='teal')
            ax7.axhline(50, color='black', linestyle='--', alpha=0.3)
            ax7.set_ylim([0, 100])
        ax7.set_title('Cumulative Win Rate (%)', fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. Metrics Summary Table
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        
        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Total P&L', f'${total_pnl:.2f}', '✓' if total_pnl > 0 else '✗'],
            ['Final Return', f'{final_return:.2f}%', '✓' if final_return > 0 else '✗'],
            ['Sharpe Ratio', f'{sharpe:.3f}', '✓' if sharpe > 0.5 else '✗'],
            ['Sortino Ratio', f'{sortino:.3f}', '✓' if sortino > 0.5 else '✗'],
            ['Max Drawdown', f'{max_dd*100:.2f}%', '✓' if max_dd < 0.15 else '✗'],
            ['Win Rate', f'{win_rate*100:.1f}%', '✓' if win_rate > 0.5 else '✗'],
            ['Total Trades', f'{num_trades}', '—'],
        ]
        
        if forced_close:
            metrics_data.append(['Exit Status', 'Forced Liquidation', '⚠️'])
        elif was_skipped:
            metrics_data.append(['Exit Status', 'Supervisor Stop', '⛔'])
        
        table = ax8.table(cellText=metrics_data, cellLoc='left',
                          colWidths=[0.3, 0.3, 0.1],
                          loc='center', bbox=[0.1, 0.0, 0.8, 1.0])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color coding rows
        for i in range(1, len(metrics_data) + 1):
            # Row index in table starts at 0 (header) -> i
            # Data index starts at 0 -> i-1
            status = metrics_data[i-1][2]
            if status == '✓':
                color = '#90EE90'
            elif status == '✗':
                color = '#FFB6C6'
            elif status == '⚠️':
                color = '#FFD700' # Gold for forced exit
            elif status == '⛔':
                color = '#FF6B6B'
            else:
                color = 'white'
                
            table[(i, 2)].set_facecolor(color)
            
        # Header style
        for j in range(3):
            table[(0, j)].set_facecolor('#4CAF50')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        # Save figure
        filename = f"{pair_name.replace('-', '_')}_analysis.png"
        filepath = os.path.join(self.output_dir, "pairs", filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved pair analysis: {filepath}")
    
    def visualize_portfolio(self, all_traces: List[Dict], skipped_pairs: List[Dict], 
                          final_summary: Dict):
        """
        Create portfolio-level aggregate visualization.
        
        Args:
            all_traces: All traces from all pairs
            skipped_pairs: List of pairs stopped by supervisor
            final_summary: Dict from supervisor.evaluate_portfolio()
        """
        # Group traces by pair
        traces_by_pair = {}
        for t in all_traces:
            pair = t['pair']
            if pair not in traces_by_pair:
                traces_by_pair[pair] = []
            traces_by_pair[pair].append(t)
        
        metrics = final_summary['metrics']
        
        # Create figure
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)
        fig.suptitle('Portfolio Aggregate Analysis', fontsize=18, fontweight='bold')
        
        # 1. Portfolio Price Performance (Base 100) - AGGREGATE BY STEP
        ax1 = fig.add_subplot(gs[0, :])
        
        # Aggregate returns by step across all pairs
        step_returns = {}  # step -> list of returns
        for pair_name, pair_traces in traces_by_pair.items():
            for trace in pair_traces:
                step = trace['local_step']
                ret = trace['daily_return']
                if step not in step_returns:
                    step_returns[step] = []
                step_returns[step].append(ret)
        
        # Average returns per step
        sorted_steps = sorted(step_returns.keys())
        avg_returns = [np.mean(step_returns[s]) for s in sorted_steps]
        
        # Convert to price performance (base 100)
        cum_returns = np.cumprod([1 + r for r in avg_returns])
        price_performance = 100 * cum_returns
        final_performance = ((price_performance[-1] / 100) - 1) * 100 if len(price_performance) > 0 else 0
        
        # Count total portfolio steps (matches the graph)
        total_portfolio_steps = len(sorted_steps)
        
        ax1.plot(sorted_steps, price_performance, linewidth=2.5, color='darkblue', 
                label='Portfolio Value')
        ax1.axhline(100, color='black', linestyle='--', alpha=0.3, label='Starting Value (100)')
        ax1.fill_between(sorted_steps, 100, price_performance, alpha=0.3,
                        color='blue' if final_performance > 0 else 'red')
        ax1.set_title(f"Portfolio Price Performance: {final_performance:.2f}% | "
                     f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
                     f"Sortino: {metrics['sortino_ratio']:.2f}",
                     fontsize=13, fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Portfolio Value (Base 100)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Per-Pair Performance Comparison
        ax2 = fig.add_subplot(gs[1, :2])
        pair_summaries = metrics['pair_summaries']
        pair_names = [p['pair'] for p in pair_summaries]
        pair_returns = [p['cum_return'] * 100 for p in pair_summaries]
        colors = ['green' if r > 0 else 'red' for r in pair_returns]
        
        # Mark skipped pairs
        for i, name in enumerate(pair_names):
            if any(skip['pair'] == name for skip in skipped_pairs):
                colors[i] = 'orange'
        
        bars = ax2.barh(pair_names, pair_returns, color=colors, alpha=0.7)
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_title('Per-Pair Returns (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Return (%)')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, pair_returns)):
            label = f'{val:.1f}%'
            if any(skip['pair'] == pair_names[i] for skip in skipped_pairs):
                label += ' ⛔'
            ax2.text(val, bar.get_y() + bar.get_height()/2, label,
                    ha='left' if val > 0 else 'right', va='center', fontsize=9)
        
        # 3. Risk Metrics Summary
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.axis('off')
        
        risk_data = [
            ['Risk Metric', 'Value'],
            ['Max Drawdown', f"{metrics['max_drawdown']*100:.2f}%"],
            ['VaR (95%)', f"{metrics['var_95']:.4f}"],
            ['CVaR (95%)', f"{metrics['cvar_95']:.4f}"],
            ['Volatility', f"{metrics['std_return']:.4f}"],
            ['', ''],
            ['Performance', ''],
            ['Win Rate', f"{metrics['win_rate']*100:.1f}%"],
            ['Avg Return/Trade', f"{metrics['avg_return']:.5f}"],
            ['Pairs Traded', f"{metrics['n_pairs']}"],
            ['Pairs Stopped', f"{len(skipped_pairs)}"],
        ]
        
        table = ax3.table(cellText=risk_data, cellLoc='left',
                         colWidths=[0.6, 0.4],
                         loc='center', bbox=[0.0, 0.0, 1.0, 1.0])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        
        # Style
        for i in range(2):
            table[(0, i)].set_facecolor('#FF6B6B')
            table[(0, i)].set_text_props(weight='bold', color='white')
        table[(6, 0)].set_facecolor('#4CAF50')
        table[(6, 0)].set_text_props(weight='bold', color='white')
        
        # 4. Sharpe Ratio by Pair
        ax4 = fig.add_subplot(gs[2, 0])
        pair_sharpes = [p['sharpe'] for p in pair_summaries]
        colors_sharpe = ['green' if s > 0.5 else 'red' if s < 0 else 'orange' for s in pair_sharpes]
        ax4.barh(pair_names, pair_sharpes, color=colors_sharpe, alpha=0.7)
        ax4.axvline(0, color='black', linestyle='-', linewidth=1)
        ax4.axvline(0.5, color='green', linestyle='--', alpha=0.3, label='Good (>0.5)')
        ax4.set_title('Sharpe Ratio by Pair', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Sharpe Ratio')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Sortino Ratio by Pair
        ax5 = fig.add_subplot(gs[2, 1])
        pair_sortinos = [p['sortino'] for p in pair_summaries]
        colors_sortino = ['green' if s > 0.5 else 'red' if s < 0 else 'orange' for s in pair_sortinos]
        ax5.barh(pair_names, pair_sortinos, color=colors_sortino, alpha=0.7)
        ax5.axvline(0, color='black', linestyle='-', linewidth=1)
        ax5.axvline(0.5, color='green', linestyle='--', alpha=0.3, label='Good (>0.5)')
        ax5.set_title('Sortino Ratio by Pair', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Sortino Ratio')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. Max Drawdown by Pair
        ax6 = fig.add_subplot(gs[2, 2])
        pair_dds = [p['max_drawdown'] * 100 for p in pair_summaries]
        colors_dd = ['red' if dd > 15 else 'orange' if dd > 10 else 'green' for dd in pair_dds]
        ax6.barh(pair_names, pair_dds, color=colors_dd, alpha=0.7)
        ax6.axvline(15, color='red', linestyle='--', alpha=0.3, label='Risk Limit')
        ax6.set_title('Max Drawdown by Pair (%)', fontsize=11, fontweight='bold')
        ax6.set_xlabel('Drawdown (%)')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3, axis='x')
        
        # 7. Portfolio Returns Distribution (ONLY TRADES - NON-ZERO)
        ax7 = fig.add_subplot(gs[3, 0])
        all_returns = [t['daily_return'] for t in all_traces if t['daily_return'] != 0]
        if all_returns:
            ax7.hist(all_returns, bins=60, alpha=0.7, color='steelblue', edgecolor='black')
            ax7.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
            ax7.axvline(np.mean(all_returns), color='orange', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(all_returns):.5f}')
            ax7.legend()
        ax7.set_title('Portfolio Trade Returns Distribution (Non-Zero Only)', 
                     fontsize=11, fontweight='bold')
        ax7.set_xlabel('Return')
        ax7.set_ylabel('Frequency')
        ax7.grid(True, alpha=0.3)
        
        # 8. Sharpe vs Return Scatter
        ax8 = fig.add_subplot(gs[3, 1])
        for pair_summary in pair_summaries:
            sharpe = pair_summary['sharpe']
            final_ret = pair_summary['cum_return'] * 100
            color = 'orange' if any(s['pair'] == pair_summary['pair'] for s in skipped_pairs) else 'blue'
            marker = 'X' if any(s['pair'] == pair_summary['pair'] for s in skipped_pairs) else 'o'
            ax8.scatter(sharpe, final_ret, s=100, alpha=0.6, color=color, marker=marker)
        
        ax8.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax8.axvline(0, color='black', linestyle='--', alpha=0.3)
        ax8.set_title('Sharpe vs Return', fontsize=11, fontweight='bold')
        ax8.set_xlabel('Sharpe Ratio')
        ax8.set_ylabel('Final Return (%)')
        ax8.grid(True, alpha=0.3)
        
        # 9. Supervisor Interventions Timeline
        ax9 = fig.add_subplot(gs[3, 2])
        if skipped_pairs:
            # Use concatenated step positions
            skip_steps = []
            current_step = 0
            for pair_name in sorted(traces_by_pair.keys()):
                pair_traces = traces_by_pair[pair_name]
                if any(skip['pair'] == pair_name for skip in skipped_pairs):
                    skip_steps.append(current_step + len(pair_traces))
                current_step += len(pair_traces)
            
            skip_names = [skip['pair'].split('-')[0][:4] for skip in skipped_pairs]
            skip_severities = [skip.get('severity', 'unknown') for skip in skipped_pairs]
            colors_severity = ['red' if s == 'critical' else 'orange' for s in skip_severities]
            
            ax9.scatter(skip_steps, range(len(skip_steps)), s=200, c=colors_severity, 
                       marker='X', alpha=0.7)
            for i, (step, name) in enumerate(zip(skip_steps, skip_names)):
                ax9.text(step, i, f' {name}', va='center', fontsize=9)
            
            ax9.set_title('Supervisor Interventions', fontsize=11, fontweight='bold')
            ax9.set_xlabel('Concatenated Step')
            ax9.set_yticks([])
            ax9.grid(True, alpha=0.3, axis='x')
        else:
            ax9.text(0.5, 0.5, 'No Interventions', ha='center', va='center', 
                    fontsize=14, transform=ax9.transAxes)
            ax9.set_title('Supervisor Interventions', fontsize=11, fontweight='bold')
            ax9.axis('off')
        
        # 10. Summary Text
        ax10 = fig.add_subplot(gs[4, :])
        ax10.axis('off')
        
        summary_text = f"""PORTFOLIO SUMMARY:
                        • Total P&L: ${metrics['total_pnl']:.2f} | Sharpe: {metrics['sharpe_ratio']:.2f} | Sortino: {metrics['sortino_ratio']:.2f}
                        • Win Rate: {metrics['win_rate']*100:.1f}% | Max Drawdown: {metrics['max_drawdown']*100:.2f}% | Total Steps: {total_portfolio_steps}
                        • Pairs Traded: {metrics['n_pairs']} | Pairs Stopped by Supervisor: {len(skipped_pairs)}
                        
                        SUPERVISOR ACTIONS:
                        """
        if final_summary['actions']:
            for action in final_summary['actions']:
                severity_symbol = "🔴" if action.get('severity') == 'high' else "🟡"
                summary_text += f"{severity_symbol} {action['action']}: {action['reason']}\n"
        else:
            summary_text += "✅ No risk interventions required\n"
        
        ax10.text(0.05, 0.5, summary_text, fontsize=11, verticalalignment='center',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"portfolio_analysis_{timestamp}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Saved portfolio analysis: {filepath}")
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio using CONFIG risk-free rate."""
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
        """Calculate Sortino ratio using CONFIG risk-free rate."""
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


def generate_all_visualizations(all_traces: List[Dict], 
                                skipped_pairs: List[Dict],
                                final_summary: Dict,
                                output_dir: str = "reports"):
    """
    Generate complete visual report for portfolio and all pairs.
    
    Args:
        all_traces: All traces from operator
        skipped_pairs: List of skipped pair info from supervisor
        final_summary: Dict from supervisor.evaluate_portfolio()
        output_dir: Where to save reports
    """
    print("\n" + "="*70)
    print("GENERATING VISUAL REPORTS")
    print("="*70)
    
    visualizer = PortfolioVisualizer(output_dir)
    
    # Group traces by pair
    traces_by_pair = {}
    for t in all_traces:
        pair = t['pair']
        if pair not in traces_by_pair:
            traces_by_pair[pair] = []
        traces_by_pair[pair].append(t)
    
    # Generate individual pair reports
    print("\n📊 Generating pair-level reports...")
    for pair_name, traces in traces_by_pair.items():
        # Check if this pair was skipped
        skip_info = next((s for s in skipped_pairs if s['pair'] == pair_name), None)
        was_skipped = skip_info is not None
        
        visualizer.visualize_pair(traces, pair_name, was_skipped, skip_info)
    
    # Generate portfolio aggregate
    print("\n📊 Generating portfolio aggregate report...")
    visualizer.visualize_portfolio(all_traces, skipped_pairs, final_summary)
    
    print(f"\n✅ All reports saved to: {output_dir}/")
    print(f"   • Portfolio aggregate: {output_dir}/portfolio_analysis_*.png")
    print(f"   • Individual pairs: {output_dir}/pairs/*.png")
