# ============================================================
# RUN A STRATEGY:  python3 run_backtest.py strategies/btc_pnp.json
# Options:  --demo (synthetic data)   --no-null (skip GBM test)
# Outputs:  console report, <name>_trades.csv, <name>_report.png
# ============================================================

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine import (load_data, generate_signals, apply_filters, simulate,
                    metrics, print_metrics, ema)
from quant_models import monte_carlo, markov_regimes, gbm_null_test, kelly


def run_strategy_on_df(df, spec):
    sigs = generate_signals(df, spec)
    sigs = apply_filters(df, sigs, spec)
    return simulate(df, sigs, spec)


def main():
    args = sys.argv[1:]
    demo = '--demo' in args
    do_null = '--null' in args
    spec_path = next((a for a in args if a.endswith('.json')), None)
    if not spec_path:
        print("Usage: python3 run_backtest.py strategies/<spec>.json [--demo] [--no-null]")
        sys.exit(1)

    spec = json.load(open(spec_path))
    name = spec['name'].replace(' ', '_').lower()
    print("=" * 60)
    print(f"  STRATEGY: {spec['name']}")
    print(f"  Interval {spec['interval']} | RR 1:{spec.get('risk_reward', 2)} | "
          f"Patterns: {', '.join(spec.get('patterns', []))}")
    print("=" * 60)

    all_trades, dfs = [], {}
    for sym in spec['symbols']:
        df = load_data(sym, spec['interval'], spec.get('period_days', 55), demo=demo)
        dfs[sym] = df
        t = run_strategy_on_df(df, spec)
        if len(t):
            t['symbol'] = sym
            all_trades.append(t)
        print_metrics(metrics(t), f"{sym}")

    if not all_trades:
        print("\nNo trades generated. Loosen filters or extend period_days.")
        sys.exit(0)

    trades = pd.concat(all_trades).sort_values('time').reset_index(drop=True)
    m = metrics(trades)
    print_metrics(m, "★ COMBINED (all symbols)")

    print("\n  BY PATTERN:")
    for pat, sub in trades.groupby('pattern'):
        print(f"    {pat:<22} {len(sub):>4} trades | win {sub['R'].gt(0).mean()*100:4.0f}% "
              f"| avg R {sub['R'].mean():+.2f}")

    # ── QUANT MODEL LAYER ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  QUANT MODEL LAYER")
    print("=" * 60)

    mc = monte_carlo(trades['R'])
    if mc:
        print("\n  MONTE CARLO (5,000 reshuffles of trade order):")
        print(f"    Median max drawdown : {mc['median_max_dd_R']:.1f} R")
        print(f"    Worst 5% drawdown   : {mc['worst5pct_dd_R']:.1f} R")
        print(f"    P(end in loss)      : {mc['p_lose_overall']*100:.1f}%")
        print(f"    Final R 90% range   : {mc['final_R_5pct']:+.1f} … {mc['final_R_95pct']:+.1f}")

    first_sym = spec['symbols'][0]
    mk = markov_regimes(dfs[first_sym]['Close'])
    print(f"\n  MARKOV REGIME MODEL ({first_sym}, 2 states: calm/volatile):")
    print(f"    P(calm→calm)        : {mk['P'][0,0]*100:.0f}%   "
          f"P(volatile→volatile): {mk['P'][1,1]*100:.0f}%")
    print(f"    Expected calm run   : {mk['expected_bars_calm']:.0f} bars | "
          f"volatile run: {mk['expected_bars_volatile']:.0f} bars")
    st = mk['state'].reindex(trades[trades.symbol == first_sym]['time']).dropna()
    if len(st) > 5:
        sub = trades[trades.symbol == first_sym].set_index('time')
        for lbl, code in [('calm', 0), ('volatile', 1)]:
            rs = sub.loc[st[st == code].index, 'R']
            if len(rs) >= 3:
                print(f"    Avg R in {lbl:<9}: {rs.mean():+.2f}  ({len(rs)} trades)")

    k = kelly(m['win_rate'], spec.get('risk_reward', 2))
    print(f"\n  KELLY CRITERION:")
    print(f"    Full Kelly risk/trade: {k*100:.1f}% of capital "
          f"(practical: {k*25:.1f}% = quarter-Kelly)")

    if do_null:
        print(f"\n  GBM NULL TEST (Black-Scholes price model, no patterns):")
        print("    Running strategy on 10 synthetic GBM paths...")
        null_R, mu, sigma = gbm_null_test(dfs[first_sym], spec, run_strategy_on_df)
        if len(null_R):
            beat = (m['avg_R'] > null_R).mean() * 100
            print(f"    GBM paths avg R     : {null_R.mean():+.2f} "
                  f"(range {null_R.min():+.2f} … {null_R.max():+.2f})")
            print(f"    Real data avg R     : {m['avg_R']:+.2f} → beats {beat:.0f}% of random paths")
            print("    → Edge likely REAL ✓" if beat >= 90 else
                  "    → Edge NOT distinguishable from luck ✗")
        else:
            print("    Not enough trades on GBM paths to compare.")

    # ── CHARTS ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    eq = trades['R'].cumsum()
    axes[0, 0].plot(eq.values, color='royalblue', linewidth=1.8)
    axes[0, 0].set_title(f"{spec['name']} — Equity Curve (R multiples)")
    axes[0, 0].set_xlabel('Trade #'); axes[0, 0].set_ylabel('Cumulative R')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(trades['R'], bins=25, color='seagreen', edgecolor='black')
    axes[0, 1].axvline(0, color='red', linewidth=1)
    axes[0, 1].set_title('R Distribution per Trade'); axes[0, 1].grid(alpha=0.3)

    if mc:
        rng = np.random.default_rng(7)
        for _ in range(200):
            axes[1, 0].plot(rng.permutation(trades['R'].values).cumsum(),
                            alpha=0.05, color='grey')
        axes[1, 0].plot(eq.values, color='royalblue', linewidth=2, label='Actual')
        axes[1, 0].set_title('Monte Carlo — 200 Alternate Trade Orderings')
        axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    price = dfs[first_sym]['Close']
    colors = np.where(mk['state'].values == 1, 'lightcoral', 'lightgreen')
    axes[1, 1].scatter(range(len(price)), price.values, c=colors, s=1)
    axes[1, 1].plot(ema(price, spec.get('ema', 200)).values, color='black', linewidth=1)
    axes[1, 1].set_title(f'{first_sym} — Markov Regimes (green=calm, red=volatile) + EMA')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{name}_report.png', dpi=150)
    trades.to_csv(f'{name}_trades.csv', index=False)
    print(f"\n✓ Saved: {name}_report.png, {name}_trades.csv")


if __name__ == '__main__':
    main()
