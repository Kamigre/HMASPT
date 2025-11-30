import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from datetime import datetime

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
    
    def visualize_pair(self, traces: List[Dict], pair_name: str, 
                       was_skipped: bool = False, skip_info: Dict = None):
        """
        Create detailed visualization for a single pair.
        
        Args:
            traces: List of trace dicts for this pair
            pair_name: Name like "AAPL-MSFT"
            was_skipped: Whether supervisor stopped this pair early
            skip_info: Dict with skip details if applicable
        """
        if len(traces) == 0:
            return
        
        # Extract data
        steps = [t['step'] for t in traces]
        pnls = [t['realized_pnl_this_step'] for t in traces]
        returns = [t['daily_return'] for t in traces]
        cum_return = [t['cum_return'] for t in traces]
        positions = [t['position'] for t in traces]
        drawdowns = [t['max_drawdown'] for t in traces]
        
        # Calculate metrics
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        total_pnl = sum(pnls)
        final_return = cum_return[-1] * 100
        max_dd = max(drawdowns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Title with skip indicator
        title = f"Pair Trading Analysis: {pair_name}"
        if was_skipped:
            title += f" ⛔ STOPPED BY SUPERVISOR"
        fig.suptitle(title, fontsize=16, fontweight='bold', 
                     color='red' if was_skipped else 'black')
        
        # 1. Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(steps, np.array(cum_return) * 100, linewidth=2, color='darkblue')
        ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax1.fill_between(steps, 0, np.array(cum_return) * 100, 
                         alpha=0.3, color='blue' if final_return > 0 else 'red')
        
        if was_skipped and skip_info:
            skip_step = skip_info.get('step_stopped', steps[-1])
            ax1.axvline(skip_step, color='red', linestyle='--', linewidth=2, 
                       label=f'Supervisor Stop: {skip_info.get("reason", "")}')
            ax1.legend(loc='upper left')
        
        ax1.set_title(f'Cumulative Return: {cum_return[-1]*100:.2f}%', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.fill_between(steps, 0, np.array(drawdowns) * 100, 
                         alpha=0.5, color='red')
        ax2.plot(steps, np.array(drawdowns) * 100, linewidth=2, color='darkred')
        ax2.set_title(f'Drawdown (Max: {max_dd*100:.2f}%)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        # 3. Daily P&L
        ax3 = fig.add_subplot(gs[1, 1])
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax3.bar(steps, pnls, color=colors, alpha=0.6)
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title(f'Daily P&L (Total: ${total_pnl:.2f})', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('P&L ($)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Position Over Time
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(steps, positions, linewidth=2, color='purple', alpha=0.7)
        ax4.fill_between(steps, 0, positions, alpha=0.3, color='purple')
        ax4.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax4.set_title('Position Over Time', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Position Size')
        ax4.grid(True, alpha=0.3)
        
        # 5. Returns Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax5.axvline(np.mean(returns), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(returns):.4f}')
        ax5.axvline(np.median(returns), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(returns):.4f}')
        ax5.set_title('Returns Distribution', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Return')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Rolling Sharpe (30-step window)
        ax6 = fig.add_subplot(gs[2, 1])
        if len(returns) > 30:
            rolling_sharpe = pd.Series(returns).rolling(30).apply(
                lambda x: self._calculate_sharpe(x.tolist())
            )
            ax6.plot(steps, rolling_sharpe, linewidth=2, color='darkgreen')
            ax6.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax6.axhline(1, color='green', linestyle=':', alpha=0.5, label='Good (>1)')
            ax6.axhline(-1, color='red', linestyle=':', alpha=0.5, label='Poor (<-1)')
            ax6.set_title(f'Rolling Sharpe (30-step)', fontsize=11, fontweight='bold')
            ax6.set_xlabel('Step')
            ax6.set_ylabel('Sharpe Ratio')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Win Rate Over Time
        ax7 = fig.add_subplot(gs[2, 2])
        window = 20
        rolling_wr = pd.Series([1 if r > 0 else 0 for r in returns]).rolling(window).mean()
        ax7.plot(steps, rolling_wr * 100, linewidth=2, color='teal')
        ax7.axhline(50, color='black', linestyle='--', alpha=0.3, label='50%')
        ax7.fill_between(steps, 50, rolling_wr * 100, 
                         alpha=0.3, color='green', where=(rolling_wr * 100 > 50))
        ax7.fill_between(steps, 50, rolling_wr * 100, 
                         alpha=0.3, color='red', where=(rolling_wr * 100 <= 50))
        ax7.set_title(f'Rolling Win Rate ({window}-step)', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Step')
        ax7.set_ylabel('Win Rate (%)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0, 100])
        
        # 8. Metrics Summary Table
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Total P&L', f'${total_pnl:.2f}', '✓' if total_pnl > 0 else '✗'],
            ['Final Return', f'{final_return:.2f}%', '✓' if final_return > 0 else '✗'],
            ['Sharpe Ratio', f'{sharpe:.3f}', '✓' if sharpe > 0.5 else '✗'],
            ['Sortino Ratio', f'{sortino:.3f}', '✓' if sortino > 0.5 else '✗'],
            ['Max Drawdown', f'{max_dd*100:.2f}%', '✓' if max_dd < 0.15 else '✗'],
            ['Win Rate', f'{win_rate*100:.1f}%', '✓' if win_rate > 0.5 else '✗'],
            ['Total Steps', f'{len(traces)}', '—'],
            ['Avg Return/Step', f'{np.mean(returns):.5f}', '—'],
        ]
        
        if was_skipped and skip_info:
            metrics_data.append(['Supervisor Action', 'STOPPED EARLY', '⛔'])
            metrics_data.append(['Stop Reason', skip_info.get('reason', 'N/A'), '—'])
        
        table = ax8.table(cellText=metrics_data, cellLoc='left', 
                         colWidths=[0.3, 0.3, 0.1],
                         loc='center', bbox=[0.1, 0.0, 0.8, 1.0])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code status
        for i in range(1, len(metrics_data)):
            if metrics_data[i][2] == '✓':
                table[(i, 2)].set_facecolor('#90EE90')
            elif metrics_data[i][2] == '✗':
                table[(i, 2)].set_facecolor('#FFB6C6')
            elif metrics_data[i][2] == '⛔':
                table[(i, 2)].set_facecolor('#FF6B6B')
        
        # Save figure
        filename = f"{pair_name.replace('-', '_')}_analysis.png"
        filepath = os.path.join(self.output_dir, "pairs", filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Saved pair analysis: {filepath}")
    
    def visualize_portfolio(self, all_traces: List[Dict], 
                           skipped_pairs: List[Dict],
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
        
        # 1. Portfolio Cumulative Returns
        ax1 = fig.add_subplot(gs[0, :])
        portfolio_cum_pnl = np.cumsum([t['realized_pnl_this_step'] for t in all_traces])
        steps = list(range(len(all_traces)))
        ax1.plot(steps, portfolio_cum_pnl, linewidth=2.5, color='darkblue', label='Portfolio')
        ax1.fill_between(steps, 0, portfolio_cum_pnl, alpha=0.3, color='blue')
        ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
        
        # Mark supervisor interventions
        for skip in skipped_pairs:
            skip_traces = traces_by_pair.get(skip['pair'], [])
            if skip_traces:
                skip_step = skip_traces[-1]['step']
                ax1.axvline(skip_step, color='red', linestyle=':', alpha=0.5)
        
        ax1.set_title(f"Portfolio P&L: ${metrics['total_pnl']:.2f} | "
                     f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
                     f"Sortino: {metrics['sortino_ratio']:.2f}",
                     fontsize=13, fontweight='bold')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Cumulative P&L ($)')
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
            ['Avg Return', f"{metrics['avg_return']:.5f}"],
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
        colors_sharpe = ['green' if s > 0.5 else 'red' if s < 0 else 'orange' 
                        for s in pair_sharpes]
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
        colors_sortino = ['green' if s > 0.5 else 'red' if s < 0 else 'orange' 
                         for s in pair_sortinos]
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
        colors_dd = ['red' if dd > 15 else 'orange' if dd > 10 else 'green' 
                    for dd in pair_dds]
        ax6.barh(pair_names, pair_dds, color=colors_dd, alpha=0.7)
        ax6.axvline(15, color='red', linestyle='--', alpha=0.3, label='Risk Limit')
        ax6.set_title('Max Drawdown by Pair (%)', fontsize=11, fontweight='bold')
        ax6.set_xlabel('Drawdown (%)')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3, axis='x')
        
        # 7. Portfolio Returns Distribution
        ax7 = fig.add_subplot(gs[3, 0])
        all_returns = [t['daily_return'] for t in all_traces]
        ax7.hist(all_returns, bins=60, alpha=0.7, color='steelblue', edgecolor='black')
        ax7.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax7.axvline(np.mean(all_returns), color='orange', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(all_returns):.5f}')
        ax7.set_title('Portfolio Returns Distribution', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Return')
        ax7.set_ylabel('Frequency')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Win Rate Scatter
        ax8 = fig.add_subplot(gs[3, 1])
        for pair_summary in pair_summaries:
            pair_traces = traces_by_pair[pair_summary['pair']]
            pair_rets = [t['cum_return'] for t in pair_traces]
            win_rate = sum(1 for r in pair_rets if r > 0) / len(pair_rets)
            final_ret = pair_summary['cum_return'] * 100
            
            color = 'orange' if any(s['pair'] == pair_summary['pair'] for s in skipped_pairs) else 'blue'
            ax8.scatter(win_rate * 100, final_ret, s=100, alpha=0.6, color=color)
        
        ax8.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax8.axvline(50, color='black', linestyle='--', alpha=0.3)
        ax8.set_title('Win Rate vs Final Return', fontsize=11, fontweight='bold')
        ax8.set_xlabel('Win Rate (%)')
        ax8.set_ylabel('Final Return (%)')
        ax8.grid(True, alpha=0.3)
        
        # 9. Supervisor Interventions Timeline
        ax9 = fig.add_subplot(gs[3, 2])
        if skipped_pairs:
            skip_steps = [skip['step_stopped'] for skip in skipped_pairs]
            skip_names = [skip['pair'].split('-')[0][:4] for skip in skipped_pairs]
            ax9.scatter(skip_steps, range(len(skip_steps)), s=200, 
                       color='red', marker='X', alpha=0.7)
            for i, (step, name) in enumerate(zip(skip_steps, skip_names)):
                ax9.text(step, i, f' {name}', va='center', fontsize=9)
            ax9.set_title('Supervisor Interventions', fontsize=11, fontweight='bold')
            ax9.set_xlabel('Step')
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
        
        summary_text = f"""
                      PORTFOLIO SUMMARY:
                      • Total P&L: ${metrics['total_pnl']:.2f}  |  Sharpe: {metrics['sharpe_ratio']:.2f}  |  Sortino: {metrics['sortino_ratio']:.2f}
                      • Win Rate: {metrics['win_rate']*100:.1f}%  |  Max Drawdown: {metrics['max_drawdown']*100:.2f}%  |  Total Steps: {metrics['total_steps']}
                      • Pairs Traded: {metrics['n_pairs']}  |  Pairs Stopped by Supervisor: {len(skipped_pairs)}
                      
                      SUPERVISOR ACTIONS:
                      """
        if final_summary['actions']:
            for action in final_summary['actions']:
                summary_text += f"• {action['action']}: {action['reason']}\n"
        else:
            summary_text += "• No risk interventions required\n"
        
        ax10.text(0.05, 0.5, summary_text, fontsize=11, verticalalignment='center',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"portfolio_analysis_{timestamp}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Saved portfolio analysis: {filepath}")
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        rf_daily = 0.04 / 252
        excess = np.array(returns) - rf_daily
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)
        if std_excess < 1e-8:
            return 0.0
        return (mean_excess / std_excess) * np.sqrt(252)
    
    def _calculate_sortino(self, returns: List[float]) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        rf_daily = 0.04 / 252
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
