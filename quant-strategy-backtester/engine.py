# ============================================================
# UNIVERSAL STRATEGY BACKTEST ENGINE
# Feed it any strategy as a JSON spec (converted from a PDF or
# Pine Script) → it returns full backtest data.
#
# Pattern library: double top/bottom, head & shoulders,
#   box breakout, level traps, order blocks (FVG)
# Filters: session window, EMA proximity, candle size, max SL
# Quant layer: Monte Carlo, Markov regimes, GBM null test, Kelly
# ============================================================

import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

INTERVAL_MIN = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}


# ============================================================
# DATA
# ============================================================

def load_data(symbol, interval='15m', period_days=55, demo=False):
    """Intraday OHLC from Yahoo Finance. Note Yahoo limits:
    5m/15m → ~60 days history, 1h → ~730 days. Falls back to
    synthetic random-walk data when offline (demo)."""
    if not demo:
        try:
            df = yf.download(symbol, period=f'{period_days}d', interval=interval,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close']].dropna()
            if len(df) > 300:
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                print(f"  ✓ {symbol}: {len(df)} live {interval} bars")
                return df
            raise ValueError('insufficient data')
        except Exception as e:
            print(f"  ⚠ {symbol}: live download failed — using synthetic data")

    step = INTERVAL_MIN[interval]
    n = int(period_days * 24 * 60 / step)
    rng = np.random.default_rng(abs(hash(symbol)) % 2**32)
    rets = rng.normal(0, 0.0015, n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.roll(close, 1); open_[0] = 100
    wick = np.abs(rng.normal(0, 0.0008, n))
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    idx = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=n, freq=f'{step}min')
    print(f"  [DEMO] {symbol}: {n} synthetic {interval} bars")
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close}, index=idx)


# ============================================================
# INDICATORS & SWINGS
# ============================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def swing_points(df, k=5):
    """Indices of local swing highs / lows (strongest in ±k bars)."""
    hi, lo = df['High'].values, df['Low'].values
    n = len(df)
    sh = [i for i in range(k, n - k) if hi[i] == hi[i - k:i + k + 1].max()]
    sl = [i for i in range(k, n - k) if lo[i] == lo[i - k:i + k + 1].min()]
    return sh, sl


# ============================================================
# PATTERN DETECTORS
# Each returns signals: dict(i, side(+1/-1), entry, sl, pattern)
# ============================================================

def detect_double_top_bottom(df, sh, sl, tol=0.003, max_gap=80, confirm=30):
    hi, lo, cl = df['High'].values, df['Low'].values, df['Close'].values
    sigs = []
    for a, b in zip(sh, sh[1:]):                      # double top → short
        if not (5 <= b - a <= max_gap):
            continue
        p1, p2 = hi[a], hi[b]
        if abs(p1 - p2) / p1 > tol:
            continue
        neck = lo[a:b + 1].min()
        if (min(p1, p2) - neck) / p1 < 0.0015:        # need a real V in the middle
            continue
        for i in range(b + 1, min(b + 1 + confirm, len(df))):
            if cl[i] < neck:
                sigs.append(dict(i=i, side=-1, entry=cl[i], sl=max(p1, p2),
                                 pattern='double_top')); break
            if hi[i] > max(p1, p2):
                break
    for a, b in zip(sl, sl[1:]):                      # double bottom → long
        if not (5 <= b - a <= max_gap):
            continue
        p1, p2 = lo[a], lo[b]
        if abs(p1 - p2) / p1 > tol:
            continue
        neck = hi[a:b + 1].max()
        if (neck - max(p1, p2)) / p1 < 0.0015:
            continue
        for i in range(b + 1, min(b + 1 + confirm, len(df))):
            if cl[i] > neck:
                sigs.append(dict(i=i, side=1, entry=cl[i], sl=min(p1, p2),
                                 pattern='double_bottom')); break
            if lo[i] < min(p1, p2):
                break
    return sigs


def detect_head_shoulders(df, sh, sl, tol=0.01, confirm=30):
    hi, lo, cl = df['High'].values, df['Low'].values, df['Close'].values
    sigs = []
    for a, b, c in zip(sh, sh[1:], sh[2:]):           # H&S → short
        head, ls, rs = hi[b], hi[a], hi[c]
        if head > ls and head > rs and abs(ls - rs) / head < tol \
                and (head - max(ls, rs)) / head > 0.0015:
            neck = lo[a:c + 1].min()
            for i in range(c + 1, min(c + 1 + confirm, len(df))):
                if cl[i] < neck:
                    sigs.append(dict(i=i, side=-1, entry=cl[i], sl=rs,
                                     pattern='head_shoulders')); break
                if hi[i] > head:
                    break
    for a, b, c in zip(sl, sl[1:], sl[2:]):           # inverse H&S → long
        head, ls, rs = lo[b], lo[a], lo[c]
        if head < ls and head < rs and abs(ls - rs) / head < tol \
                and (min(ls, rs) - head) / head > 0.0015:
            neck = hi[a:c + 1].max()
            for i in range(c + 1, min(c + 1 + confirm, len(df))):
                if cl[i] > neck:
                    sigs.append(dict(i=i, side=1, entry=cl[i], sl=rs,
                                     pattern='inv_head_shoulders')); break
                if lo[i] < head:
                    break
    return sigs


def detect_box_breakout(df, interval, box_hours=4, box_pct=0.6, min_touch=2):
    """Consolidation box of `box_hours` with ≥2 touches each side;
    trade the close beyond the box."""
    hi, lo, cl = df['High'].values, df['Low'].values, df['Close'].values
    w = max(8, int(box_hours * 60 / INTERVAL_MIN[interval]))
    sigs, i = [], w
    while i < len(df):
        h, l = hi[i - w:i].max(), lo[i - w:i].min()
        rng_ = h - l
        if rng_ > 0 and rng_ / cl[i - 1] <= box_pct / 100:
            top_t = (hi[i - w:i] >= h - 0.15 * rng_).sum()
            bot_t = (lo[i - w:i] <= l + 0.15 * rng_).sum()
            if top_t >= min_touch and bot_t >= min_touch:
                if cl[i] > h:
                    sigs.append(dict(i=i, side=1, entry=cl[i], sl=l, pattern='box_breakout'))
                    i += w; continue
                if cl[i] < l:
                    sigs.append(dict(i=i, side=-1, entry=cl[i], sl=h, pattern='box_breakout'))
                    i += w; continue
        i += 1
    return sigs


def detect_order_block(df, lookforward=100):
    """BIG R logic: Fair Value Gap (gap between candle-1 high and
    candle-3 low) marks candle-1 as an Order Block; trade the retest."""
    hi, lo, cl = df['High'].values, df['Low'].values, df['Close'].values
    sigs = []
    for i in range(2, len(df) - 1):
        if lo[i] > hi[i - 2]:                          # bullish FVG
            top, bot = hi[i - 2], lo[i - 2]
            for j in range(i + 1, min(i + lookforward, len(df))):
                if lo[j] <= top:                       # retest of OB
                    sigs.append(dict(i=j, side=1, entry=cl[j], sl=bot,
                                     pattern='bullish_order_block'))
                    break
        if hi[i] < lo[i - 2]:                          # bearish FVG
            top, bot = hi[i - 2], lo[i - 2]
            for j in range(i + 1, min(i + lookforward, len(df))):
                if hi[j] >= bot:
                    sigs.append(dict(i=j, side=-1, entry=cl[j], sl=top,
                                     pattern='bearish_order_block'))
                    break
    return sigs


def detect_trap_level(df, level_mode='prev_day_high_low', tz='UTC'):
    """Trap = price wicks through a key level but closes back inside →
    enter the reversal. Levels: previous day's high/low, or the first
    4H candle's high/low (Trap Trading strategy)."""
    local = df.index.tz_convert(tz)
    dates = pd.Series(local.date, index=df.index)
    hi, lo, cl = df['High'].values, df['Low'].values, df['Close'].values
    sigs = []
    for day, day_idx in dates.groupby(dates).groups.items():
        pos = [df.index.get_loc(t) for t in day_idx]
        if level_mode == 'prev_day_high_low':
            prev_mask = dates < day
            if not prev_mask.any():
                continue
            prev_day = dates[prev_mask].iloc[-1]
            ppos = [df.index.get_loc(t) for t in dates[dates == prev_day].index]
            lvl_h, lvl_l = hi[ppos].max(), lo[ppos].min()
            scan = pos
        else:                                          # first_4h_candle
            t0 = local[pos[0]]
            first = [p for p in pos if (local[p] - t0).total_seconds() < 4 * 3600]
            if len(first) < 2 or len(first) >= len(pos):
                continue
            lvl_h, lvl_l = hi[first].max(), lo[first].min()
            scan = pos[len(first):]
        done_h = done_l = False
        for p in scan:
            if not done_h and hi[p] > lvl_h and cl[p] < lvl_h:    # trap at high → short
                sigs.append(dict(i=p, side=-1, entry=cl[p], sl=hi[p], pattern='trap_high'))
                done_h = True
            if not done_l and lo[p] < lvl_l and cl[p] > lvl_l:    # trap at low → long
                sigs.append(dict(i=p, side=1, entry=cl[p], sl=lo[p], pattern='trap_low'))
                done_l = True
    return sigs


PATTERN_GROUP = {
    'double_top': 'double_top_bottom', 'double_bottom': 'double_top_bottom',
    'head_shoulders': 'head_shoulders', 'inv_head_shoulders': 'head_shoulders',
    'box_breakout': 'box_breakout',
    'bullish_order_block': 'order_block', 'bearish_order_block': 'order_block',
    'trap_high': 'trap_level', 'trap_low': 'trap_level',
}


def generate_signals(df, spec):
    """Run every pattern detector enabled in the spec."""
    sh, sl = swing_points(df, k=spec.get('swing_k', 5))
    sigs = []
    pats = spec.get('patterns', [])
    if 'double_top_bottom' in pats:
        sigs += detect_double_top_bottom(df, sh, sl, tol=spec.get('dt_tol', 0.003))
    if 'head_shoulders' in pats:
        sigs += detect_head_shoulders(df, sh, sl)
    if 'box_breakout' in pats:
        sigs += detect_box_breakout(df, spec['interval'],
                                    box_hours=spec.get('box_hours', 4),
                                    box_pct=spec.get('box_pct', 0.6))
    if 'order_block' in pats:
        sigs += detect_order_block(df)
    if 'trap_level' in pats:
        sigs += detect_trap_level(df, spec.get('level_mode', 'prev_day_high_low'),
                                  tz=spec.get('session', {}).get('tz', 'UTC'))
    return sorted(sigs, key=lambda s: s['i'])


# ============================================================
# FILTERS (session window, EMA location, candle size, max SL)
# ============================================================

def _in_session(ts, session):
    t = ts.tz_convert(session.get('tz', 'UTC')).strftime('%H:%M')
    if not (session.get('start', '00:00') <= t <= session.get('end', '23:59')):
        return False
    for a, b in session.get('exclude', []):
        if a <= t <= b:
            return False
    return True

def apply_filters(df, sigs, spec):
    ema_arr = ema(df['Close'], spec.get('ema', 200)).values if spec.get('ema') else None
    hi, lo, cl = df['High'].values, df['Low'].values, df['Close'].values
    out = []
    for s in sigs:
        i = s['i']
        if 'session' in spec and not _in_session(df.index[i], spec['session']):
            continue
        size = (hi[i] - lo[i]) / cl[i] * 100
        if size > spec.get('max_candle_pct', 99) or size < spec.get('min_candle_pct', 0):
            continue
        risk_pct = abs(s['entry'] - s['sl']) / s['entry'] * 100
        if risk_pct < 0.03 or risk_pct > spec.get('max_sl_pct', 99):
            continue
        if ema_arr is not None:
            dist = abs(s['entry'] - ema_arr[i]) / s['entry'] * 100
            group = PATTERN_GROUP.get(s['pattern'], s['pattern'])
            loc = spec.get('pattern_filters', {}).get(group, {}) \
                      .get('ema_location', spec.get('ema_location', 'any'))
            near = spec.get('near_ema_pct', 0.5)
            if loc == 'near' and dist > near:
                continue
            if loc == 'away' and dist <= near:
                continue
        out.append(s)
    return out


# ============================================================
# TRADE SIMULATOR (R-multiple based, breakeven shift, 1 trade at a time)
# ============================================================

def simulate(df, sigs, spec):
    rr = spec.get('risk_reward', 2.0)
    be_at = spec.get('breakeven_at', 0.5)             # shift SL→entry at 50% of target
    one_per_day = spec.get('one_trade_per_day', False)
    hi, lo, cl = df['High'].values, df['Low'].values, df['Close'].values
    trades, busy_until, taken_days = [], -1, set()
    for s in sigs:
        i = s['i']
        if i <= busy_until:
            continue
        day = df.index[i].date()
        if one_per_day and day in taken_days:
            continue
        side, entry, sl0 = s['side'], s['entry'], s['sl']
        risk = abs(entry - sl0)
        tp = entry + side * rr * risk
        be_trig = entry + side * be_at * rr * risk
        cur_sl, result, j = sl0, None, i
        mae = 0.0   # max adverse excursion: worst point against us (in R)
        mfe = 0.0   # max favorable excursion: best point for us (in R)
        for j in range(i + 1, len(df)):
            if side == 1:
                mae = min(mae, (lo[j] - entry) / risk)
                mfe = max(mfe, (hi[j] - entry) / risk)
                if lo[j] <= cur_sl:
                    result = (cur_sl - entry) / risk; break
                if hi[j] >= tp:
                    result = rr; break
                if hi[j] >= be_trig:
                    cur_sl = max(cur_sl, entry)
            else:
                mae = min(mae, (entry - hi[j]) / risk)
                mfe = max(mfe, (entry - lo[j]) / risk)
                if hi[j] >= cur_sl:
                    result = (entry - cur_sl) / risk; break
                if lo[j] <= tp:
                    result = rr; break
                if lo[j] <= be_trig:
                    cur_sl = min(cur_sl, entry)
        if result is None:
            result = side * (cl[-1] - entry) / risk    # mark-to-market at data end
        trades.append(dict(time=df.index[i], pattern=s['pattern'],
                           side='LONG' if side == 1 else 'SHORT',
                           entry=round(entry, 4), sl=round(sl0, 4),
                           R=round(result, 2), bars_held=j - i,
                           MAE=round(mae, 2), MFE=round(mfe, 2)))
        busy_until = j
        taken_days.add(day)
    return pd.DataFrame(trades)


# ============================================================
# METRICS
# ============================================================

def metrics(trades):
    if len(trades) == 0:
        return {}
    r = trades['R']
    wins, losses = r[r > 0], r[r <= 0]
    eq = r.cumsum()
    dd = (eq - eq.cummax()).min()
    streak = mx = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        mx = max(mx, streak)
    return {
        'trades': len(r),
        'win_rate': (r > 0).mean(),
        'avg_R': r.mean(),
        'total_R': r.sum(),
        'profit_factor': wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf,
        'max_drawdown_R': dd,
        'max_consec_losses': mx,
    }

def print_metrics(m, label):
    if not m:
        print(f"  {label}: no trades")
        return
    print(f"\n  {label}")
    print(f"    Trades         : {m['trades']}")
    print(f"    Win rate       : {m['win_rate']*100:.1f}%")
    print(f"    Avg R / trade  : {m['avg_R']:+.2f}")
    print(f"    Total R        : {m['total_R']:+.1f}")
    print(f"    Profit factor  : {m['profit_factor']:.2f}")
    print(f"    Max DD (R)     : {m['max_drawdown_R']:.1f}")
    print(f"    Max consec loss: {m['max_consec_losses']}")
