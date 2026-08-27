"""
Label variant definitions for three-class labeling.

Five variants (L1–L5) each specify a target percentage, stop percentage,
and forward horizon (candles). Phase 6 trains one model per variant.
"""

LABEL_VARIANTS: dict[str, dict] = {
    "L1": {
        "target_pct":  0.0040,
        "stop_pct":    0.0020,
        "horizon":     20,
        "description": "Standard 2:1 RR — matches baseline strategy",
    },
    "L2": {
        "target_pct":  0.0060,
        "stop_pct":    0.0030,
        "horizon":     30,
        "description": "Wider 2:1 RR — fewer trades, larger moves",
    },
    "L3": {
        "target_pct":  0.0030,
        "stop_pct":    0.0030,
        "horizon":     15,
        "description": "Symmetric 1:1 RR — requires higher win rate",
    },
    "L4": {
        "target_pct":  0.0050,
        "stop_pct":    0.0025,
        "horizon":     25,
        "description": "Intermediate 2:1 RR — balanced between L1 and L2",
    },
    "L5": {
        "target_pct":  0.0200,   # +2.00% long target / -2.00% short target
        "stop_pct":    0.0100,   # -1.00% long stop   / +1.00% short stop
        "horizon":     60,        # 60 × 3min = 180 minutes; 2% moves need 2-4h
        "description": "Wide 2:1 RR — target=2.00%, stop=1.00%, horizon=60 candles",
    },
    "L6": {
        "target_pct":  0.0140,   # +1.40% long target / -1.40% short target
        "stop_pct":    0.0070,   # -0.70% long stop   / +0.70% short stop
        "horizon":     40,        # 40 × 3min = 120 minutes; sweet spot between L1 and L5
        "description": "Sweet-spot 2:1 RR — target=1.40%, stop=0.70%, horizon=40",
    },
}
