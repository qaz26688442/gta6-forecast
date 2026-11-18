"""GTA VI Launch-Day Twitch CCU Quantile Forecasting Model.

Predicts concurrent viewer distributions for hours 2..24 of launch day on Twitch.
Combines empirical comparable diurnal curve modeling with adaptive log-scale volatility.
Version: 1.1.0 - Enhanced quantile smoothing and boundary guardrails.
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Baseline fallback curve derived from major AAA release diurnal curves (hours 2..24 relative to hour 1)
DEFAULT_CURVE = [
    0.871, 0.754, 0.733, 0.747, 0.788, 0.802, 0.899, 0.882,
    0.912, 0.928, 0.978, 1.059, 1.181, 1.518, 1.652, 1.771,
    1.951, 2.402, 2.372, 2.330, 2.298, 2.182, 2.039,
]


def round2(x: float) -> float | int:
    """Rounds x to 2 decimals half-up, converting whole numbers to int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    val = whole / 100
    return int(val) if val.is_integer() else val


def extract_comparable_curves(case: dict) -> dict[int, list[float]]:
    """Extracts hour-by-hour viewer multipliers relative to hour 1 from comparable games."""
    hourly_ratios: dict[int, list[float]] = {h: [] for h in range(2, 25)}

    for comp in case.get("comparables", []):
        day_start = comp.get("day_start")
        if not day_start:
            continue
        prefix = day_start[:10]
        release_points = [h for h in comp.get("hourly", []) if h.get("from", "").startswith(prefix)]
        if len(release_points) >= 24:
            v1 = max(1, release_points[0].get("viewers", 1))
            for idx in range(1, 24):
                hour = idx + 1
                vh = release_points[idx].get("viewers", 1)
                # Robust ratio clamping to avoid extreme outlier contamination
                ratio = max(0.08, min(15.0, vh / v1))
                hourly_ratios[hour].append(ratio)

    return hourly_ratios


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} invalid levels {case.get('levels')}")

    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if not first:
        raise ValueError(f"{name} missing hour 1 print")

    p1 = max(1.0, float(first.get("viewers", 1)))
    comp_ratios = extract_comparable_curves(case)

    results = []
    for row in case.get("forecast", []):
        h = row["hour"]
        ratios = comp_ratios.get(h, [])

        if len(ratios) >= 5:
            med_ratio = statistics.median(ratios)
            log_vals = [math.log(r) for r in ratios]
            spread = statistics.stdev(log_vals) if len(log_vals) > 1 else 0.25
        else:
            med_ratio = DEFAULT_CURVE[h - 2]
            spread = 0.25

        # Refined volatility diffusion over the forecasting horizon
        eff_spread = max(0.18, min(0.60, spread * 0.82 + 0.035 * math.sqrt(h - 1)))

        quantiles = []
        for z in Z:
            q_val = p1 * med_ratio * math.exp(z * eff_spread)
            q_val = max(1.0, min(1e9, q_val))
            quantiles.append(round2(q_val))

        # Enforce non-decreasing monotonic sequence
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]

        results.append({"hour": h, "q": quantiles})

    return {"hours": results}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"Generated forecast for {path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()

