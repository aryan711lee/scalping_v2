"""
Label variant definitions for three-class labeling.

Four variants (L1–L4) each specify a target percentage, stop percentage,
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
}
