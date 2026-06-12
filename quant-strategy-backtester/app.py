# ============================================================
# QUANT STRATEGY BACKTESTER — WEB APP
# Run:  streamlit run app.py
# Tabs: Market Overview (live candlesticks) | Backtest | Add Strategy
# ============================================================

import json
import glob
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from engine import (load_data, generate_signals, apply_filters, simulate,
                    metrics, ema, INTERVAL_MIN)
from quant_models import (monte_carlo, markov_regimes, kelly,
                          fit_garch, walk_forward, walk_forward_verdict)
from tv_report import performance_summary, overview_numbers, list_of_trades

st.set_page_config(page_title="Quant Strategy Backtester", layout="wide")

GREEN, RED, ACCENT, GREY = "#26a69a", "#ef5350", "#42a5f5", "#8b95a9"

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: #161b26;
    border: 1px solid #232a3b;
    border-radius: 12px;
    padding: 14px 16px;
}
div[data-testid="stMetric"] label { color: #8b95a9; }
div[data-testid="stMetricValue"] { font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Quant Strategy Backtester")

specs = {}
for f in sorted(glob.glob('strategies/*.json')):
    try:
        specs[json.load(open(f))['name']] = f
    except Exception:
        pass

tab_mkt, tab_bt, tab_add = st.tabs(["📊 Market Overview", "🧪 Backtest", "➕ Add Strategy"])

# ============================================================
# CANDLESTICK HELPER (TradingView-style)
# ============================================================

def candle_chart(df, sym, ema_n=200, trades=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name=sym, increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
    if ema_n:
        fig.add_trace(go.Scatter(x=df.index, y=ema(df['Close'], ema_n),
                                 line=dict(color='#ff9800', width=1.5), name=f'EMA {ema_n}'))
    if trades is not None and len(trades):
        longs = trades[trades['side'] == 'LONG']
        shorts = trades[trades['side'] == 'SHORT']
        if len(longs):
            fig.add_trace(go.Scatter(x=longs['time'], y=longs['entry'], mode='markers',
                                     marker=dict(symbol='triangle-up', size=12, color='#00e676'),
                                     name='Long entry',
                                     text=[f"{p} | {r:+.1f}R" for p, r in zip(longs['pattern'], longs['R'])],
                                     hovertemplate='%{text}<br>%{x}<br>entry %{y}'))
        if len(shorts):
            fig.add_trace(go.Scatter(x=shorts['time'], y=shorts['entry'], mode='markers',
                                     marker=dict(symbol='triangle-down', size=12, color='#ff1744'),
                                     name='Short entry',
                                     text=[f"{p} | {r:+.1f}R" for p, r in zip(shorts['pattern'], shorts['R'])],
                                     hovertemplate='%{text}<br>%{x}<br>entry %{y}'))
    fig.update_layout(height=480, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10),
                      template='plotly_dark', legend=dict(orientation='h', y=1.05))
    return fig


def show_data_status(df, sym):
    """Make it obvious whether the data is LIVE or synthetic."""
    last = df.index[-1]
    is_live = (pd.Timestamp.now(tz='UTC') - last.tz_convert('UTC')).days < 7
    chg = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{sym} last price", f"{df['Close'].iloc[-1]:,.2f}", f"{chg:+.2f}%")
    c2.metric("Bars loaded", f"{len(df):,}")
    c3.metric("Last bar", last.strftime('%d %b %Y %H:%M'))
    c4.metric("Data", "🟢 LIVE" if is_live else "🔴 SYNTHETIC/STALE")
    if not is_live:
        st.error("This is NOT live data (Yahoo download failed or demo mode on). "
                 "Results below are meaningless for real markets.")


# ── universal symbol catalog: indices, stocks, forex, metals, energy, crypto ──
WATCHLIST = {
    'Indices': {'^GSPC': 'S&P 500', '^NDX': 'Nasdaq 100',
                '^NSEI': 'Nifty 50', '^DJI': 'Dow Jones'},
    'Commodities': {'GC=F': 'Gold', 'SI=F': 'Silver',
                    'CL=F': 'Crude Oil', 'NG=F': 'Natural Gas'},
    'Forex': {'EURUSD=X': 'EUR/USD', 'GBPUSD=X': 'GBP/USD',
              'USDJPY=X': 'USD/JPY', 'USDINR=X': 'USD/INR'},
    'Crypto': {'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum'},
    'Stocks': {'AAPL': 'Apple', 'TSLA': 'Tesla',
               'RELIANCE.NS': 'Reliance', 'ICICIBANK.NS': 'ICICI Bank'},
}
ALL_SYMBOLS = {f"{name}  ·  {sym}": sym
               for grp in WATCHLIST.values() for sym, name in grp.items()}


def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


@st.cache_data(ttl=900, show_spinner=False)
def market_scan(demo=False):
    """Daily-data scan of the whole watchlist: trend, momentum, vol, verdict."""
    rows = []
    for group, syms in WATCHLIST.items():
        for sym, name in syms.items():
            try:
                df = load_data(sym, '1d', 400, demo=demo)
                c = df['Close']
                if len(c) < 60:
                    continue
                last = float(c.iloc[-1])
                sma50 = float(c.rolling(50).mean().iloc[-1])
                sma200 = (float(c.rolling(200).mean().iloc[-1])
                          if len(c) >= 200 else np.nan)
                rsi14 = float(_rsi(c).iloc[-1])
                vol = float(c.pct_change().rolling(20).std().iloc[-1]
                            * np.sqrt(252) * 100)
                score = (1 if last > sma50 else -1)
                if not np.isnan(sma200):
                    score += 1 if last > sma200 else -1
                score += 1 if rsi14 > 55 else (-1 if rsi14 < 45 else 0)
                verdict = ('🟢 Bullish' if score >= 2 else
                           '🔴 Bearish' if score <= -2 else '🟡 Neutral')
                rows.append({
                    'Group': group, 'Market': name, 'Symbol': sym,
                    'Price': round(last, 4 if last < 10 else 2),
                    '1D %': round(float(c.pct_change().iloc[-1]) * 100, 2),
                    '5D %': round(float(c.pct_change(5).iloc[-1]) * 100, 2),
                    '20D %': round(float(c.pct_change(20).iloc[-1]) * 100, 2),
                    'Vol ann. %': round(vol, 1),
                    'RSI 14': int(round(rsi14)),
                    'vs 50MA': '↑ above' if last > sma50 else '↓ below',
                    'vs 200MA': (('↑ above' if last > sma200 else '↓ below')
                                 if not np.isnan(sma200) else '—'),
                    'Verdict': verdict,
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


# ============================================================
# TAB 1 — MARKET OVERVIEW
# ============================================================

with tab_mkt:
    st.caption("Cross-asset scan — indices, stocks, forex, gold, oil, crypto: "
               "trend, momentum, volatility and a verdict for every market.")
    if st.button("🔭 Scan all markets", type="primary"):
        with st.spinner("Scanning 16 markets (daily data, cached 15 min)..."):
            scan = market_scan()
        if len(scan) == 0:
            st.error("No data returned — check your internet connection.")
        else:
            bull = float((scan['Verdict'] == '🟢 Bullish').mean())
            best = scan.loc[scan['20D %'].idxmax()]
            worst = scan.loc[scan['20D %'].idxmin()]
            hot = scan.loc[scan['Vol ann. %'].idxmax()]
            mc = st.columns(4)
            mc[0].metric("Bullish breadth",
                         f"{bull*100:.0f}%",
                         f"{int((scan['Verdict']=='🟢 Bullish').sum())} of {len(scan)} markets")
            mc[1].metric("Best 20-day", best['Market'], f"{best['20D %']:+.1f}%")
            mc[2].metric("Worst 20-day", worst['Market'], f"{worst['20D %']:+.1f}%",
                         delta_color="inverse")
            mc[3].metric("Most volatile", hot['Market'], f"{hot['Vol ann. %']:.0f}% ann.")
            st.caption("Verdict = price vs 50MA + price vs 200MA + RSI tilt. "
                       "Breadth above ~60% = broad risk-on environment.")
            for grp in scan['Group'].unique():
                st.subheader(grp)
                st.dataframe(scan[scan['Group'] == grp]
                             .drop(columns='Group').set_index('Market'),
                             use_container_width=True)

    st.divider()
    st.subheader("Deep dive — single market chart")
    cols = st.columns([2, 1, 1])
    pick = cols[0].selectbox("Market", list(ALL_SYMBOLS))
    interval = cols[1].selectbox("Timeframe", ["15m", "5m", "1h", "1d"], index=0)
    days = cols[2].slider("Days of history", 5,
                          700 if interval in ('1h', '1d') else 55, 30)
    if st.button("Load chart"):
        with st.spinner("Downloading live data..."):
            df = load_data(ALL_SYMBOLS[pick], interval, days)
        show_data_status(df, ALL_SYMBOLS[pick])
        st.plotly_chart(candle_chart(df, ALL_SYMBOLS[pick]),
                        use_container_width=True)

# ============================================================
# TAB 2 — BACKTEST
# ============================================================

with tab_bt:
    if not specs:
        st.warning("No strategies found in strategies/ folder.")
    else:
        choice = st.selectbox("Strategy", list(specs))
        spec = json.load(open(specs[choice]))
        with st.expander("Strategy rules (JSON spec)"):
            st.json(spec)
        pcols = st.columns(3)
        cap = pcols[0].number_input("Initial capital ($)", 1000, 10_000_000, 100_000, 1000)
        risk_pct = pcols[1].number_input("Risk per trade (%)", 0.1, 10.0, 1.0, 0.1)
        demo = pcols[2].checkbox("Demo mode (synthetic data)", value=False)

        with st.expander("📂 Data options — interval, history, custom CSV"):
            dcols = st.columns(3)
            interval_choice = dcols[0].selectbox(
                "Interval override", ["spec default", "5m", "15m", "1h", "1d"])
            period_choice = dcols[1].number_input(
                "History days (0 = spec default)", 0, 2000, 0, 5)
            csv_file = dcols[2].file_uploader(
                "Own OHLC CSV (Date,Open,High,Low,Close)", type='csv')
            sym_override = st.multiselect(
                "Symbols override — test this strategy on ANY market "
                "(indices, stocks, forex, gold, oil, crypto)",
                list(ALL_SYMBOLS), default=[])
            st.caption("Yahoo limits 5m/15m to ~60 days of history. For longer "
                       "tests pick 1h/1d here, or upload a CSV from any vendor.")

        if st.button("▶ Run backtest", type="primary"):
            interval_use = (spec['interval'] if interval_choice == 'spec default'
                            else interval_choice)
            period_use = (int(period_choice) if period_choice
                          else spec.get('period_days', 55))
            spec_run = dict(spec)
            spec_run['interval'] = interval_use
            symbols = ([ALL_SYMBOLS[s] for s in sym_override]
                       if sym_override else spec['symbols'])
            custom_df = None
            if csv_file is not None:
                raw = pd.read_csv(csv_file)
                raw.columns = [str(c).strip().title() for c in raw.columns]
                date_col = 'Date' if 'Date' in raw.columns else raw.columns[0]
                raw[date_col] = pd.to_datetime(raw[date_col], utc=True)
                custom_df = (raw.set_index(date_col)
                                [['Open', 'High', 'Low', 'Close']]
                                .astype(float).dropna().sort_index())
                symbols = ['CUSTOM CSV']
                st.info(f"Using uploaded CSV: {len(custom_df):,} bars")

            all_trades, dfs = [], {}
            prog = st.progress(0.0, "Downloading data...")
            for n, s in enumerate(symbols):
                df = (custom_df if custom_df is not None
                      else load_data(s, interval_use, period_use, demo=demo))
                dfs[s] = df
                sigs = apply_filters(df, generate_signals(df, spec_run), spec_run)
                t = simulate(df, sigs, spec_run)
                if len(t):
                    t['symbol'] = s
                    all_trades.append(t)
                prog.progress((n + 1) / len(symbols), f"Backtested {s}")
            prog.empty()

            first = symbols[0]
            show_data_status(dfs[first], first)

            # ── data quality & market context panel ──────────
            with st.expander("🔍 Data quality & market context", expanded=False):
                d = dfs[first]
                diffs = d.index.to_series().diff().dropna()
                gaps = int((diffs > diffs.median() * 3).sum())
                ret = d['Close'].pct_change()
                bpd = (1 if interval_use == '1d'
                       else max(1, int(6.5 * 60 / INTERVAL_MIN[interval_use])))
                rv = float(ret.rolling(20).std().iloc[-1]
                           * np.sqrt(252 * bpd) * 100)
                grp = d.groupby(d.index.date)
                adr = float(((grp['High'].max() - grp['Low'].min())
                             / grp['Close'].last()).mean() * 100)
                qc = st.columns(5)
                qc[0].metric("Bars received", f"{len(d):,}")
                qc[1].metric("Gaps detected", gaps,
                             "vs median bar spacing ×3")
                qc[2].metric("Timezone", str(d.index.tz))
                qc[3].metric("Realized vol (ann.)", f"{rv:.1f}%")
                qc[4].metric("Avg daily range", f"{adr:.2f}%")
                if not demo and custom_df is None:
                    try:
                        a = d['Close'].groupby(d.index.date).last().pct_change()
                        cc = st.columns(2)
                        for m, (bsym, bname) in enumerate(
                                [('GC=F', 'Gold'), ('UUP', 'US Dollar (UUP)')]):
                            bdf = load_data(bsym, '1d', max(period_use, 60))
                            b = bdf['Close'].groupby(bdf.index.date).last().pct_change()
                            cc[m].metric(f"Correlation vs {bname}",
                                         f"{a.corr(b):+.2f}")
                    except Exception:
                        st.caption("Benchmark correlations unavailable.")

            if not all_trades:
                st.warning("No trades generated in this period. The strategy's filters are "
                           "strict (session, EMA distance, candle size) — try more "
                           "period_days or another strategy.")
                st.stop()

            trades = pd.concat(all_trades).sort_values('time').reset_index(drop=True)
            trades['time'] = pd.to_datetime(trades['time'], utc=True)
            m = metrics(trades)
            px = dfs[first]['Close']
            buy_hold = float(px.iloc[-1] / px.iloc[0] - 1)
            ov = overview_numbers(trades, cap, risk_pct)

            t_ov, t_perf, t_list, t_chart, t_models, t_quant = st.tabs(
                ["Overview", "Performance Summary", "List of Trades",
                 "Chart", "Models", "Quant Models"])

            # ── OVERVIEW (TradingView-style dashboard) ─────────
            with t_ov:
                # verdict banner — plain-English conclusion
                r_vals = trades['R'].values
                n_tr = len(r_vals)
                t_stat = (r_vals.mean() / (r_vals.std() / np.sqrt(n_tr))
                          if n_tr > 1 and r_vals.std() > 0 else 0.0)
                confirmed = abs(t_stat) > 2 and n_tr >= 50
                sig_txt = ("edge statistically confirmed (t>2, n≥50)"
                           if confirmed else "edge not statistically confirmed")
                days_span = max((trades['time'].iloc[-1] - trades['time'].iloc[0]).days, 1)
                verdict = (f"{'Profitable' if ov['net_profit'] > 0 else 'Unprofitable'}: "
                           f"{'+' if ov['net_profit'] >= 0 else '−'}${abs(ov['net_profit']):,.0f} "
                           f"({ov['net_profit_pct']*100:+.2f}%) over {days_span} days, "
                           f"max drawdown −${abs(ov['max_dd']):,.0f} — {sig_txt}.")
                if ov['net_profit'] > 0 and confirmed:
                    st.success("✅ " + verdict)
                elif ov['net_profit'] > 0:
                    st.warning("🟡 " + verdict)
                else:
                    st.error("🔻 " + verdict)

                c = st.columns(6)
                c[0].metric("Net Profit", f"${ov['net_profit']:+,.0f}",
                            f"{ov['net_profit_pct']*100:+.2f}%")
                c[1].metric("Win Rate", f"{ov['pct_profitable']*100:.1f}%",
                            f"{int((r_vals > 0).sum())} of {n_tr}")
                pf = ov['profit_factor']
                c[2].metric("Profit Factor", "∞" if np.isinf(pf) else f"{pf:.2f}",
                            "wins ÷ losses")
                c[3].metric("Max Drawdown", f"−${abs(ov['max_dd']):,.0f}",
                            f"{ov['max_dd_pct']*100:.2f}%", delta_color="inverse")
                c[4].metric("Avg Trade", f"${ov['avg_trade']:+,.0f}",
                            f"{ov['avg_trade_pct']*100:+.3f}%")
                c[5].metric("Total Trades", f"{n_tr:,}",
                            f"~{ov['avg_bars']:.0f} bars held")

                # donut row
                st.caption("Where the results come from — outcomes, patterns, symbols")
                d1, d2, d3 = st.columns(3)
                donut_layout = dict(template='plotly_dark', height=290,
                                    margin=dict(l=10, r=10, t=42, b=10),
                                    showlegend=True,
                                    legend=dict(orientation='h', y=-0.12))
                wins_n = int((r_vals > 0).sum())
                loss_n = int((r_vals < 0).sum())
                be_n = n_tr - wins_n - loss_n
                with d1:
                    fig = go.Figure(go.Pie(
                        labels=['Wins', 'Losses', 'Breakeven'],
                        values=[wins_n, loss_n, be_n], hole=0.55,
                        marker=dict(colors=[GREEN, RED, GREY]),
                        textinfo='percent', sort=False))
                    fig.update_layout(title='Trade Outcomes', **donut_layout)
                    st.plotly_chart(fig, use_container_width=True)
                with d2:
                    gp = trades[trades['R'] > 0].groupby('pattern')['R'].sum()
                    if len(gp):
                        fig = go.Figure(go.Pie(
                            labels=gp.index.tolist(), values=gp.values, hole=0.55,
                            marker=dict(colors=[GREEN, ACCENT, '#ffa726',
                                                '#ab47bc', '#26c6da', '#d4e157']),
                            textinfo='percent'))
                        fig.update_layout(title='Profit by Pattern (gross wins)',
                                          **donut_layout)
                        st.plotly_chart(fig, use_container_width=True)
                with d3:
                    sym_n = trades['symbol'].value_counts()
                    fig = go.Figure(go.Pie(
                        labels=sym_n.index.tolist(), values=sym_n.values, hole=0.55,
                        marker=dict(colors=[ACCENT, GREEN, '#ffa726',
                                            '#ab47bc', '#26c6da', '#d4e157',
                                            '#ec407a', '#8d6e63']),
                        textinfo='percent'))
                    fig.update_layout(title='Trades by Symbol', **donut_layout)
                    st.plotly_chart(fig, use_container_width=True)

                # equity + drawdown
                st.caption("Equity journey — every trade's cumulative effect on capital")
                eq = ov['equity']
                eqfig = go.Figure()
                eqfig.add_trace(go.Scatter(
                    y=eq, mode='lines', name='Equity',
                    line=dict(color=GREEN, width=2.5),
                    fill='tozeroy', fillcolor='rgba(38,166,154,0.12)'))
                eqfig.add_hline(y=cap, line_dash='dot', line_color=GREY,
                                annotation_text='Initial capital')
                eqfig.update_layout(title='Equity curve ($)', height=340,
                                    template='plotly_dark', hovermode='x unified',
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    yaxis_range=[min(eq.min(), cap)*0.995,
                                                 max(eq.max(), cap)*1.005])
                st.plotly_chart(eqfig, use_container_width=True)
                ddfig = go.Figure(go.Scatter(
                    y=eq - np.maximum.accumulate(eq), mode='lines', name='Drawdown',
                    fill='tozeroy', fillcolor='rgba(239,83,80,0.25)',
                    line=dict(color=RED, width=1.5)))
                ddfig.update_layout(title='Drawdown ($)', height=200,
                                    template='plotly_dark', hovermode='x unified',
                                    margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(ddfig, use_container_width=True)

            # ── PERFORMANCE SUMMARY (All / Long / Short) ──────
            with t_perf:
                st.dataframe(performance_summary(trades, cap, risk_pct, buy_hold),
                             height=820, use_container_width=True)

            # ── LIST OF TRADES ────────────────────────────────
            with t_list:
                lot = list_of_trades(trades, cap, risk_pct)
                st.dataframe(lot, height=500, use_container_width=True)
                st.download_button("Export trades CSV", lot.to_csv(), "trades.csv")

            # ── CHART (entries marked on candles) ─────────────
            with t_chart:
                st.plotly_chart(candle_chart(dfs[first], first, spec.get('ema', 200),
                                             trades[trades['symbol'] == first]),
                                use_container_width=True)
                st.subheader("By pattern")
                st.dataframe(trades.groupby('pattern')['R']
                             .agg(trades='count',
                                  win_rate=lambda r: f"{(r>0).mean()*100:.0f}%",
                                  avg_R='mean', total_R='sum').round(2))

                if 'MAE' in trades.columns:
                    st.subheader("MAE vs final R — are the stops too tight?")
                    st.caption("MAE = worst point against the trade before it closed. "
                               "Winning trades with deep MAE nearly got stopped out — "
                               "a cluster of those means the stop is too tight.")
                    sc = go.Figure(go.Scatter(
                        x=trades['MAE'], y=trades['R'], mode='markers',
                        marker=dict(color=[GREEN if r > 0 else RED
                                           for r in trades['R']],
                                    size=9, opacity=0.75),
                        text=trades['pattern'],
                        hovertemplate='%{text}<br>MAE %{x:.2f}R → final %{y:.2f}R'))
                    sc.add_vline(x=-1.0, line_dash='dot', line_color=GREY,
                                 annotation_text='stop level (−1R)')
                    sc.update_layout(xaxis_title='MAE (R)', yaxis_title='Final R',
                                     height=340, template='plotly_dark',
                                     margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(sc, use_container_width=True)

                st.subheader("Average R by entry hour — which session earns?")
                tz_ = spec.get('session', {}).get('tz', 'UTC')
                hours = trades['time'].dt.tz_convert(tz_).dt.hour
                byh = trades.groupby(hours)['R'].agg(['mean', 'count'])
                hb = go.Figure(go.Bar(
                    x=byh.index, y=byh['mean'],
                    marker_color=[GREEN if v > 0 else RED for v in byh['mean']],
                    text=[f"{c}" for c in byh['count']],
                    hovertemplate='Hour %{x}: avg %{y:.2f}R (%{text} trades)'))
                hb.update_layout(xaxis_title=f'Entry hour ({tz_})',
                                 yaxis_title='Avg R', height=300,
                                 template='plotly_dark',
                                 margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(hb, use_container_width=True)

            # ── MODELS: GARCH + WALK-FORWARD ──────────────────
            with t_models:
                st.subheader("GARCH(1,1) volatility model")
                st.caption("σ²(t) = ω + α·r²(t−1) + β·σ²(t−1) — volatility clusters. "
                           "When forecasted vol is above its long-run level, "
                           "risk less per trade.")
                g = fit_garch(dfs[first]['Close'].pct_change())
                if g is None:
                    st.info("Need at least 100 bars to fit GARCH.")
                else:
                    gc1 = st.columns(5)
                    gc1[0].metric("α (shock)", f"{g['alpha']:.3f}")
                    gc1[1].metric("β (memory)", f"{g['beta']:.3f}")
                    gc1[2].metric("Persistence α+β", f"{g['persistence']:.3f}")
                    gc1[3].metric("Next-bar vol forecast",
                                  f"{g['forecast_vol']*100:.3f}%")
                    gc1[4].metric("Long-run vol", f"{g['long_run_vol']*100:.3f}%")
                    ret_ = dfs[first]['Close'].pct_change().dropna()
                    realized = ret_.rolling(20).std()
                    gfig = go.Figure()
                    gfig.add_trace(go.Scatter(x=ret_.index, y=g['cond_vol'] * 100,
                                              name='GARCH conditional vol',
                                              line=dict(color=ACCENT, width=1.5)))
                    gfig.add_trace(go.Scatter(x=realized.index, y=realized * 100,
                                              name='Realized vol (20-bar)',
                                              line=dict(color=GREY, width=1,
                                                        dash='dot')))
                    gfig.update_layout(title='Per-bar volatility (%)', height=320,
                                       template='plotly_dark', hovermode='x unified',
                                       margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(gfig, use_container_width=True)
                    if g['forecast_vol'] > g['long_run_vol']:
                        st.warning("Forecasted vol is ABOVE its long-run level → "
                                   "size positions smaller than usual right now.")
                    else:
                        st.success("Forecasted vol is at/below its long-run level → "
                                   "normal position sizing is acceptable.")

                st.subheader("Walk-forward analysis (4 sequential folds)")
                st.caption("A real edge shows up in every period. One great fold "
                           "and three flat ones = overfit backtest.")
                wf = walk_forward(dfs[first], spec_run, run_strategy_on_df, 4)
                if wf:
                    wfdf = pd.DataFrame(wf)
                    wfig = go.Figure(go.Bar(
                        x=[f"Fold {r['fold']}<br>{r['start']} – {r['end']}"
                           for r in wf],
                        y=wfdf['total_R'],
                        marker_color=[GREEN if v > 0 else RED
                                      for v in wfdf['total_R']],
                        text=[f"{v:+.1f}R / {t} trades"
                              for v, t in zip(wfdf['total_R'], wfdf['trades'])],
                        textposition='outside'))
                    wfig.update_layout(title='Total R per fold', height=360,
                                       template='plotly_dark',
                                       margin=dict(l=10, r=10, t=40, b=30))
                    st.plotly_chart(wfig, use_container_width=True)
                    st.info(walk_forward_verdict(wf))
                else:
                    st.info("Not enough data for walk-forward folds.")

            # ── QUANT MODELS ──────────────────────────────────
            with t_quant:
                mc = monte_carlo(trades['R'])
                mk = markov_regimes(dfs[first]['Close'])
                k = kelly(m['win_rate'], spec.get('risk_reward', 2))
                risk_amt = cap * risk_pct / 100
                q = st.columns(4)
                if mc:
                    q[0].metric("MC median max DD",
                                f"${abs(mc['median_max_dd_R'])*risk_amt:,.0f}")
                    q[1].metric("MC worst-5% DD",
                                f"${abs(mc['worst5pct_dd_R'])*risk_amt:,.0f}")
                q[2].metric("P(calm→calm)", f"{mk['P'][0,0]*100:.0f}%")
                q[3].metric("Kelly risk/trade", f"{k*100:.1f}% (use ÷4)")
                rng = np.random.default_rng(7)
                mcfig = go.Figure()
                for _ in range(150):
                    mcfig.add_trace(go.Scatter(
                        y=cap + rng.permutation(trades['R'].values).cumsum()*risk_amt,
                        mode='lines', line=dict(color='grey', width=0.5),
                        opacity=0.08, showlegend=False))
                mcfig.add_trace(go.Scatter(y=ov['equity'], mode='lines',
                                           line=dict(color='#26a69a', width=2),
                                           name='Actual'))
                mcfig.update_layout(title='Monte Carlo — alternate trade orderings ($)',
                                    height=350, template='plotly_dark',
                                    margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(mcfig, use_container_width=True)

# ============================================================
# TAB 3 — ADD A NEW STRATEGY
# ============================================================

with tab_add:
    st.markdown("**Three ways to add a strategy — it appears in the Backtest tab instantly.**")

    st.subheader("1 · Upload a JSON spec")
    up = st.file_uploader("Drop a .json spec file", type='json')
    if up:
        try:
            new_spec = json.load(up)
            path = os.path.join('strategies', up.name)
            json.dump(new_spec, open(path, 'w'), indent=2)
            st.success(f"Saved → strategies/{up.name}. Switch to the Backtest tab.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    st.subheader("2 · Build one with a form")
    with st.form("builder"):
        name = st.text_input("Strategy name", "My Strategy")
        symbols = st.text_input("Symbols (comma separated, Yahoo tickers)", "GC=F")
        interval = st.selectbox("Timeframe", ["5m", "15m", "1h", "1d"], index=1)
        pats = st.multiselect("Patterns", ["double_top_bottom", "head_shoulders",
                                           "box_breakout", "order_block", "trap_level"],
                              default=["double_top_bottom"])
        col = st.columns(4)
        rr = col[0].number_input("Risk:Reward", 1.0, 5.0, 2.0, 0.5)
        max_sl = col[1].number_input("Max SL %", 0.1, 5.0, 1.0, 0.1)
        ema_n = col[2].number_input("EMA", 0, 400, 200, 50)
        ema_loc = col[3].selectbox("EMA location", ["any", "near", "away"])
        sess = st.columns(3)
        tz = sess[0].text_input("Timezone", "Asia/Kolkata")
        t_start = sess[1].text_input("Session start", "09:15")
        t_end = sess[2].text_input("Session end", "22:00")
        if st.form_submit_button("Create strategy"):
            new_spec = {
                "name": name, "symbols": [s.strip() for s in symbols.split(',')],
                "interval": interval, "period_days": 700 if interval in ('1h', '1d') else 55,
                "patterns": pats, "risk_reward": rr, "max_sl_pct": max_sl,
                "breakeven_at": 0.5,
                "session": {"tz": tz, "start": t_start, "end": t_end},
            }
            if ema_n:
                new_spec["ema"] = int(ema_n)
                new_spec["ema_location"] = ema_loc
                new_spec["near_ema_pct"] = 0.5
            fname = name.lower().replace(' ', '_') + '.json'
            json.dump(new_spec, open(os.path.join('strategies', fname), 'w'), indent=2)
            st.success(f"Saved → strategies/{fname}. Switch to the Backtest tab.")

    st.subheader("3 · From a PDF or Pine Script")
    st.markdown("A script can't read a discretionary PDF by itself — convert it once with "
                "Claude: paste the prompt below + your PDF, get back a JSON spec, then "
                "upload it in option 1. Takes under a minute.")
    if os.path.exists('../ADD_NEW_STRATEGY_PROMPT.md'):
        st.code(open('../ADD_NEW_STRATEGY_PROMPT.md').read(), language='markdown')
    else:
        st.code(open('ADD_NEW_STRATEGY_PROMPT.md').read() if os.path.exists('ADD_NEW_STRATEGY_PROMPT.md')
                else "See ADD_NEW_STRATEGY_PROMPT.md in the repo.", language='markdown')
