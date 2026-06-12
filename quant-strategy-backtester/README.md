# Quant Strategy Backtester

A config-driven backtesting engine for intraday price-action strategies
on **gold, BTC, and forex pairs**. Any strategy is a small JSON file —
no code changes needed to test a new one.

## What it does

1. Downloads live intraday data (Yahoo Finance)
2. Detects chart patterns: double tops/bottoms, head & shoulders,
   box breakouts, level traps, order blocks
3. Applies the strategy's filters: trading session, distance from
   200 EMA, candle size, max stop-loss
4. Simulates every trade in R-multiples with realistic stop-loss,
   take-profit (1:2), and breakeven shift
5. Validates with quant models:
   - **Monte Carlo** — 5,000 reshuffles of trade order → drawdown risk
   - **Markov regime-switching** — calm vs volatile states, and which
     one the strategy actually earns in
   - **Kelly criterion** — optimal risk per trade

## Run it

```bash
pip install -r requirements.txt
python run_backtest.py strategies/pnp_gold.json
```

Output: console report + `*_report.png` (equity curve, R distribution,
Monte Carlo cone, regime map) + `*_trades.csv`.

## Strategies included

| File | Market | Logic |
|---|---|---|
| `strategies/pnp_gold.json` | Gold (GC=F) | Double top/bottom away from 200 EMA, H&S near EMA, 15m |
| `strategies/btc_pnp.json` | BTC-USD | Patterns near 200 EMA + 4h box breakout, session-filtered |
| `strategies/pnp_forex.json` | EUR, GBP, JPY | Patterns near 200 EMA, tight stops |

## Fully automated (GitHub Actions)

Every push — and every day at 08:00 IST — GitHub automatically runs
all strategies on fresh market data and saves the reports as
downloadable artifacts. See the **Actions** tab.

## Design choices

- **R-multiples instead of $ PnL** — results are position-size independent
- **Signals shifted to next bar** — no lookahead bias
- **Breakeven stop at 50% of target** — as the original strategies specify
- **One position at a time** — realistic for a discretionary-style system

## Limitations (deliberate honesty)

- Yahoo limits 5m/15m history to ~60 days → small samples; the engine
  also supports `"interval": "1h", "period_days": 700` for more trades
- Discretionary rules ("clear reaction", "definitive candle") are
  approximated mechanically — treat results as a lower bound
