# Prompt: convert any strategy PDF / Pine Script to a spec

Paste everything below into Claude along with your PDF or Pine Script.

---

Convert the attached trading strategy (PDF or Pine Script) into a JSON
spec for my backtest engine. Output ONLY the JSON.

Schema:
{
  "name": "...",
  "symbols": ["Yahoo Finance tickers, e.g. BTC-USD, GC=F, CL=F, EURUSD=X, RELIANCE.NS"],
  "interval": "5m | 15m | 1h | 1d",
  "period_days": 55,
  "patterns": ["any of: double_top_bottom, head_shoulders, box_breakout, order_block, trap_level"],
  "level_mode": "prev_day_high_low | first_4h_candle   (only for trap_level)",
  "ema": 200,
  "near_ema_pct": 0.5,
  "ema_location": "near | away | any",
  "pattern_filters": {"<pattern>": {"ema_location": "near|away|any"}},
  "box_hours": 4, "box_pct": 0.6,
  "session": {"tz": "Asia/Kolkata", "start": "10:00", "end": "22:00", "exclude": [["16:00","18:00"]]},
  "min_candle_pct": 0, "max_candle_pct": 99,
  "max_sl_pct": 1.0,
  "risk_reward": 2.0,
  "breakeven_at": 0.5,
  "one_trade_per_day": false
}

Rules:
- Map the strategy's entry patterns to the closest pattern(s) in the library.
- Convert all times to the session block with the right timezone.
- If the strategy has a rule the schema can't express, add a "note" field
  describing what was approximated.
- If it needs a completely new pattern, say so and describe the detection
  logic so I can add it to engine.py.
