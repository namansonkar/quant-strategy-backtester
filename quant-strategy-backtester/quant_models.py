# ============================================================
# QUANT MODEL LAYER
# The "is this edge real?" toolkit:
#   1. Monte Carlo  — reshuffle trade sequence → drawdown risk
#   2. Markov chain — regime-switching model of the market
#   3. GBM null test — Black-Scholes price model as the
#      "no-edge" benchmark: if the strategy profits on random
#      GBM paths too, the patterns mean nothing
#   4. Kelly criterion — optimal risk per trade
# ============================================================

import numpy as np
import pandas as pd


def monte_carlo(trade_R, n_sims=5000, seed=7):
    """Reshuffle the order of trade outcomes thousands of times.
    The trades are the same — only the SEQUENCE changes — which
    reveals the distribution of drawdowns you could face."""
    r = np.asarray(trade_R)
    if len(r) < 5:
        return None
    rng = np.random.default_rng(seed)
    dds, finals = [], []
    for _ in range(n_sims):
        path = rng.permutation(r).cumsum()
        dds.append((path - np.maximum.accumulate(path)).min())
        finals.append(path[-1])
    dds, finals = np.array(dds), np.array(finals)
    return {
        'median_max_dd_R': np.median(dds),
        'worst5pct_dd_R': np.percentile(dds, 5),
        'p_lose_overall': (finals < 0).mean(),
        'final_R_5pct': np.percentile(finals, 5),
        'final_R_95pct': np.percentile(finals, 95),
    }


def markov_regimes(close, vol_window=50):
    """2-state Markov regime-switching model (calm vs volatile).
    States from rolling volatility; we estimate the transition
    matrix P and each regime's persistence — the same idea as
    Hamilton's regime-switching models, kept transparent."""
    ret = close.pct_change()
    vol = ret.rolling(vol_window).std()
    state = (vol > vol.median()).astype(int)          # 0 = calm, 1 = volatile
    s = state.dropna().values
    P = np.zeros((2, 2))
    for a, b in zip(s[:-1], s[1:]):
        P[a, b] += 1
    P = P / P.sum(axis=1, keepdims=True)
    persistence = [1 / (1 - P[i, i]) if P[i, i] < 1 else np.inf for i in (0, 1)]
    return {
        'P': P,                       # P[i,j] = prob of moving from state i to j
        'expected_bars_calm': persistence[0],
        'expected_bars_volatile': persistence[1],
        'state': pd.Series(state, index=close.index),
    }


def gbm_paths(close, n_paths=10, seed=11):
    """Geometric Brownian Motion — the price model underlying
    Black-Scholes. Same drift & volatility as the real series,
    but ZERO exploitable patterns. Used as the null hypothesis."""
    ret = close.pct_change().dropna()
    mu, sigma = ret.mean(), ret.std()
    rng = np.random.default_rng(seed)
    n = len(close)
    paths = []
    for _ in range(n_paths):
        r = rng.normal(mu, sigma, n)
        paths.append(close.iloc[0] * np.cumprod(1 + r))
    return paths, mu, sigma


def gbm_null_test(df, spec, run_strategy_fn, n_paths=10):
    """Run the FULL strategy on synthetic GBM paths. If the real
    avg R beats ~all GBM runs, the edge comes from real structure
    in prices, not from the mechanics of the rules."""
    paths, mu, sigma = gbm_paths(df['Close'], n_paths)
    null_avg_R = []
    for p in paths:
        fake = df.copy()
        scale = p / df['Close'].values
        for col in ['Open', 'High', 'Low', 'Close']:
            fake[col] = df[col].values * scale
        t = run_strategy_fn(fake, spec)
        if len(t) >= 5:
            null_avg_R.append(t['R'].mean())
    return np.array(null_avg_R), mu, sigma


def kelly(win_rate, rr):
    """Kelly criterion: optimal fraction of capital to risk.
    f* = p − (1−p)/b. Pros use quarter- to half-Kelly."""
    if rr <= 0:
        return 0.0
    return max(0.0, win_rate - (1 - win_rate) / rr)


# ============================================================
# GARCH(1,1) — volatility clustering model
# σ²(t) = ω + α·r²(t−1) + β·σ²(t−1)
# Fitted by maximum likelihood with scipy. Volatility forecasts
# drive position sizing: risk less when forecasted vol is high.
# ============================================================

def fit_garch(returns):
    """Fit GARCH(1,1) on a return series via MLE.
    Returns dict with params, conditional vol series, next-step forecast."""
    from scipy.optimize import minimize

    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) < 100:
        return None
    r = r - r.mean()
    var_uncond = r.var()

    def neg_loglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10
        sig2 = np.empty(len(r))
        sig2[0] = var_uncond
        for t in range(1, len(r)):
            sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
        sig2 = np.maximum(sig2, 1e-12)
        return 0.5 * np.sum(np.log(sig2) + r ** 2 / sig2)

    x0 = np.array([var_uncond * 0.05, 0.08, 0.90])
    res = minimize(neg_loglik, x0, method='Nelder-Mead',
                   options={'maxiter': 2000})
    omega, alpha, beta = res.x

    sig2 = np.empty(len(r))
    sig2[0] = var_uncond
    for t in range(1, len(r)):
        sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1]
    cond_vol = np.sqrt(np.maximum(sig2, 1e-12))
    next_var = omega + alpha * r[-1] ** 2 + beta * sig2[-1]
    persistence = alpha + beta
    return {
        'omega': float(omega), 'alpha': float(alpha), 'beta': float(beta),
        'persistence': float(persistence),         # near 1 = vol shocks last long
        'cond_vol': cond_vol,                      # per-bar conditional volatility
        'forecast_vol': float(np.sqrt(max(next_var, 0))),
        'long_run_vol': float(np.sqrt(omega / max(1 - persistence, 1e-6))),
        'converged': bool(res.success),
    }


# ============================================================
# WALK-FORWARD ANALYSIS — the honest robustness test
# Split data into sequential folds; a real edge shows up in
# (almost) every fold. One great fold + three flat ones = overfit.
# ============================================================

def walk_forward(df, spec, run_strategy_fn, n_folds=4):
    """Run the full strategy per sequential fold."""
    n = len(df)
    fold_size = n // n_folds
    results = []
    for k in range(n_folds):
        chunk = df.iloc[k * fold_size:(k + 1) * fold_size]
        if len(chunk) < 100:
            continue
        t = run_strategy_fn(chunk, spec)
        results.append({
            'fold': k + 1,
            'start': chunk.index[0].strftime('%d %b %y'),
            'end': chunk.index[-1].strftime('%d %b %y'),
            'trades': len(t),
            'total_R': float(t['R'].sum()) if len(t) else 0.0,
            'avg_R': float(t['R'].mean()) if len(t) else 0.0,
        })
    return results


def walk_forward_verdict(results):
    """Plain-English robustness verdict from fold results."""
    if not results:
        return "Not enough data for walk-forward analysis."
    pos = sum(1 for r in results if r['total_R'] > 0)
    n = len(results)
    if pos == n:
        return f"✅ Profitable in all {n} folds — consistent, low overfitting risk."
    if pos >= n - 1:
        return f"🟡 Profitable in {pos} of {n} folds — promising but not uniform."
    if pos > 0:
        return (f"⚠️ Profitable in only {pos} of {n} folds — performance is "
                "concentrated in one period: likely overfit or regime-dependent.")
    return f"🔻 Unprofitable in all {n} folds — no evidence of edge."
