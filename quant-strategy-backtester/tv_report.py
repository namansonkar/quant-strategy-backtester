# ============================================================
# TradingView-style Strategy Tester report
# Replicates the Performance Summary / Overview metrics that
# TradingView shows, computed from this engine's R-based trades.
# $ figures assume: initial capital + fixed % risk per trade.
# ============================================================

import numpy as np
import pandas as pd


def _fmt_money(v):
    return f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"


def _fmt_pct(v):
    return f"{v*100:.2f}%"


def _side_stats(sub, cap, risk_amt):
    """One column of TradingView's Performance Summary."""
    if len(sub) == 0:
        return {}
    pnl = sub['R'].values * risk_amt
    eq = cap + np.cumsum(pnl)
    runup = float((eq - np.minimum.accumulate(eq)).max())
    drawdown = float((eq - np.maximum.accumulate(eq)).min())
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_p = float(wins.sum())
    gross_l = float(losses.sum())
    net = gross_p + gross_l
    ret = pnl / cap
    sharpe = float(ret.mean() / ret.std()) if ret.std() > 0 else 0.0
    neg = ret[ret < 0]
    sortino = float(ret.mean() / neg.std()) if len(neg) > 1 and neg.std() > 0 else 0.0
    bars = sub['bars_held'].values
    win_mask = pnl > 0
    return {
        'Net Profit': f"{_fmt_money(net)}  ({_fmt_pct(net/cap)})",
        'Gross Profit': f"{_fmt_money(gross_p)}  ({_fmt_pct(gross_p/cap)})",
        'Gross Loss': f"{_fmt_money(gross_l)}  ({_fmt_pct(gross_l/cap)})",
        'Max Equity Run-up': f"{_fmt_money(runup)}  ({_fmt_pct(runup/cap)})",
        'Max Equity Drawdown': f"{_fmt_money(drawdown)}  ({_fmt_pct(drawdown/cap)})",
        'Profit Factor': f"{gross_p/abs(gross_l):.3f}" if gross_l != 0 else "∞",
        'Sharpe Ratio (per-trade)': f"{sharpe:.3f}",
        'Sortino Ratio (per-trade)': f"{sortino:.3f}",
        'Total Closed Trades': f"{len(pnl)}",
        'Number Winning Trades': f"{int(win_mask.sum())}",
        'Number Losing Trades': f"{int((~win_mask).sum())}",
        'Percent Profitable': _fmt_pct(win_mask.mean()),
        'Avg Trade': f"{_fmt_money(pnl.mean())}  ({_fmt_pct(pnl.mean()/cap)})",
        'Avg Winning Trade': _fmt_money(wins.mean()) if len(wins) else "—",
        'Avg Losing Trade': _fmt_money(losses.mean()) if len(losses) else "—",
        'Ratio Avg Win / Avg Loss': (f"{wins.mean()/abs(losses.mean()):.3f}"
                                     if len(wins) and len(losses) and losses.mean() != 0 else "—"),
        'Largest Winning Trade': _fmt_money(wins.max()) if len(wins) else "—",
        'Largest Losing Trade': _fmt_money(losses.min()) if len(losses) else "—",
        'Avg # Bars in Trades': f"{bars.mean():.0f}",
        'Avg # Bars in Winning Trades': f"{bars[win_mask].mean():.0f}" if win_mask.any() else "—",
        'Avg # Bars in Losing Trades': f"{bars[~win_mask].mean():.0f}" if (~win_mask).any() else "—",
    }


METRIC_ORDER = [
    'Net Profit', 'Gross Profit', 'Gross Loss',
    'Max Equity Run-up', 'Max Equity Drawdown',
    'Buy & Hold Return',
    'Profit Factor', 'Sharpe Ratio (per-trade)', 'Sortino Ratio (per-trade)',
    'Total Closed Trades', 'Number Winning Trades', 'Number Losing Trades',
    'Percent Profitable',
    'Avg Trade', 'Avg Winning Trade', 'Avg Losing Trade',
    'Ratio Avg Win / Avg Loss',
    'Largest Winning Trade', 'Largest Losing Trade',
    'Avg # Bars in Trades', 'Avg # Bars in Winning Trades', 'Avg # Bars in Losing Trades',
]


def performance_summary(trades, cap=100_000.0, risk_pct=1.0, buy_hold_ret=None):
    """TradingView's Performance Summary table: All / Long / Short columns."""
    risk_amt = cap * risk_pct / 100.0
    cols = {
        'All': _side_stats(trades, cap, risk_amt),
        'Long': _side_stats(trades[trades['side'] == 'LONG'], cap, risk_amt),
        'Short': _side_stats(trades[trades['side'] == 'SHORT'], cap, risk_amt),
    }
    if buy_hold_ret is not None:
        bh = f"{_fmt_money(buy_hold_ret*cap)}  ({_fmt_pct(buy_hold_ret)})"
        cols['All']['Buy & Hold Return'] = bh
    rows = []
    for mname in METRIC_ORDER:
        if mname in cols['All']:
            rows.append({'Metric': mname,
                         'All': cols['All'].get(mname, '—'),
                         'Long': cols['Long'].get(mname, '—'),
                         'Short': cols['Short'].get(mname, '—')})
    return pd.DataFrame(rows).set_index('Metric')


def overview_numbers(trades, cap=100_000.0, risk_pct=1.0):
    """TradingView's Overview tab headline numbers."""
    risk_amt = cap * risk_pct / 100.0
    pnl = trades['R'].values * risk_amt
    eq = cap + np.cumsum(pnl)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    net = float(pnl.sum())
    return {
        'net_profit': net, 'net_profit_pct': net / cap,
        'total_trades': len(pnl),
        'pct_profitable': float((pnl > 0).mean()),
        'profit_factor': float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf,
        'max_dd': dd, 'max_dd_pct': dd / cap,
        'avg_trade': float(pnl.mean()), 'avg_trade_pct': float(pnl.mean() / cap),
        'avg_bars': float(trades['bars_held'].mean()),
        'equity': eq,
    }


def list_of_trades(trades, cap=100_000.0, risk_pct=1.0):
    """TradingView's List of Trades tab."""
    risk_amt = cap * risk_pct / 100.0
    t = trades.copy().reset_index(drop=True)
    t.index = t.index + 1
    t.index.name = 'Trade #'
    t['Profit $'] = (t['R'] * risk_amt).round(2)
    t['Profit %'] = (t['R'] * risk_amt / cap * 100).round(3)
    t['Cum. Profit $'] = t['Profit $'].cumsum().round(2)
    t = t.rename(columns={'time': 'Date/Time', 'symbol': 'Symbol', 'side': 'Side',
                          'pattern': 'Signal', 'entry': 'Entry Price', 'sl': 'Stop',
                          'R': 'R Multiple', 'bars_held': '# Bars'})
    order = ['Date/Time', 'Symbol', 'Side', 'Signal', 'Entry Price', 'Stop',
             '# Bars', 'R Multiple', 'Profit $', 'Profit %', 'Cum. Profit $']
    return t[[c for c in order if c in t.columns]]
