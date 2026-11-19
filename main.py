"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

For every trophy, the program forecasts five quantiles (levels 0.05, 0.25,
0.5, 0.75, 0.95) of its earn rate -- the percent of the players Exophase
tracks who have earned it -- at the case's target time.

Model, per trophy, in logit space (logit(p) = ln(p / (100 - p))):

1.  Trend: the earn rate moves roughly linearly in log time since release.
    Fit a recency-weighted least-squares line of logit(rate) against
    ln(1 + hours since release) over the case's distinct reads (consecutive
    identical reads are one Exophase refresh, so they collapse to a single
    point).  The fitted slope is shrunk toward zero when few distinct reads
    exist or the fit is noisy, so a flat launch-night history forecasts
    persistence, while a real multi-day history forecasts real progress.
2.  The median at the target is the last rate's logit plus the shrunk slope
    times the log-time step to the target, capped so no trophy can jump
    more than +2.2 or fall more than -1.2 logit units, and capped by tier
    (a platinum earned only by finishing everything cannot be common a
    week after launch; a 100%-completion gold stays rare; story trophies
    stay below 97%).
3.  Structure: the platinum median can never exceed the lowest other
    trophy's median (it requires all of them), and story-chapter medians
    are non-increasing along the chapter order ending at the final mission.
4.  Spread: quantiles come from a normal on the logit scale around the
    median, with a standard deviation combining a random-walk term that
    grows with the square root of the horizon, the size of the forecast
    move, the extrapolation distance, and the trend fit's residual noise.
    A 90-minute horizon is nearly certain; a 4.5-day horizon is honestly
    wide, which is what the pinball loss rewards.

Pure Python standard library, deterministic: no randomness, no wall clock,
no network, no dict/set iteration order dependence.  Reads every ``*.in``
file in the input directory (``$MAKEFUN_INPUT_DIR`` or ``/input``) and
writes one ``<id>.out`` per case to the output directory
(``$MAKEFUN_OUTPUT_DIR`` or ``/output``).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import.  This entry is one file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Median caps (percent) by tier at the target.  A week after launch almost
# nobody has finished everything, so the platinum and the 100%-completion
# gold stay in single digits; ordinary story bronze can climb high.
TIER_CAP = {"platinum": 4.0, "gold": 45.0, "silver": 70.0, "bronze": 97.0}
COMPLETION_CAP = 6.0  # gold for attaining 100% completion

# How far a median may move from the last read, in logit units.
MAX_RISE = 2.2
MAX_FALL = 1.2

# Random-walk spread, logit units per square-root day of horizon.
WALK_SIGMA = 0.5


def clamp_pct(x):
    """Clamp a percent into the range the scorer's logit can take."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def logistic(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def hours_since_release(moment, release):
    seconds = (moment - release).total_seconds()
    return max(0.0, seconds / 3600.0)


def log_time(moment, release):
    """Log time since release: ln(1 + hours).  Growth slows as hours pile up."""
    return math.log1p(hours_since_release(moment, release))


def distinct_reads(history):
    """Collapse consecutive identical reads to their first occurrence.

    Exophase refreshes on its own schedule, so long stretches of reads are
    byte-identical; each stretch is one observation, not many.
    """
    out = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if key != prev_key:
            out.append(read)
            prev_key = key
    return out


def fit_trend(points):
    """Recency-weighted least squares of y against x.

    points: list of (x, y, weight).  Returns (intercept, slope, resid_std,
    n) with slope 0 when the x-range is too small to identify one.
    """
    n = len(points)
    if n < 2:
        return (points[0][1] if points else 0.0, 0.0, 0.0, n)
    wsum = sum(w for _, _, w in points)
    xbar = sum(w * x for x, _, w in points) / wsum
    ybar = sum(w * y for _, y, w in points) / wsum
    sxx = sum(w * (x - xbar) ** 2 for x, _, w in points)
    if sxx < 1e-9:
        return (ybar, 0.0, 0.0, n)
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in points)
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    resid2 = sum(w * (y - (intercept + slope * x)) ** 2
                 for x, y, w in points) / wsum
    return (intercept, slope, math.sqrt(max(0.0, resid2)), n)


def last_known_rate(history, idx):
    """The trophy's rate in the latest read that lists it, else None."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            return rates[idx]
    return None


def tier_cap(trophy):
    if trophy.get("tier") == "gold" and "100%" in (trophy.get("description") or ""):
        return COMPLETION_CAP
    return TIER_CAP.get(trophy.get("tier"), 90.0)


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or LEVELS
    zvals = [norm_ppf(level) for level in levels]
    horizon_days = case["horizon_minutes"] / 1440.0
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = parse_time(case["target"])
    x_target = log_time(target, release)

    reads = distinct_reads(history)
    # Recency weights: a read 12 hours old counts half as much as the newest.
    last_moment = parse_time(reads[-1]["at"]) if reads else target
    x_last = log_time(last_moment, release)

    medians = []
    spreads = []
    for idx, trophy in enumerate(trophies):
        last_rate = last_known_rate(history, idx)
        if last_rate is None:
            # Never observed: fall back to a tier prior and stay uncertain.
            prior = {"platinum": 0.5, "gold": 1.0,
                     "silver": 2.0, "bronze": 5.0}.get(
                         trophy.get("tier"), 5.0)
            medians.append(logit(prior))
            spreads.append(1.5)
            continue
        base = logit(last_rate)
        points = []
        for read in reads:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            moment = parse_time(read["at"])
            age_hours = max(0.0, (last_moment - moment).total_seconds() / 3600.0)
            weight = 0.5 ** (age_hours / 12.0)
            points.append((log_time(moment, release), logit(rates[idx]), weight))
        _a, slope, resid, npts = fit_trend(points)
        # Shrink the slope: few distinct reads or a noisy fit means trust
        # persistence more.  With one distinct read there is no slope at all.
        confidence = min(1.0, npts / 6.0) if npts >= 3 else 0.0
        if resid > 0.5:
            confidence *= 0.5 / resid if resid > 0 else 1.0
        slope = slope * (0.35 + 0.65 * confidence) if npts >= 3 else 0.0
        # Extra damping for far extrapolation in log time.
        dx = x_target - x_last
        delta = slope * dx / (1.0 + 0.15 * max(0.0, dx))
        # Never extrapolate a dilution dip far: new-player inflow slows.
        delta = max(-MAX_FALL, min(MAX_RISE, delta))
        median = base + delta
        # Tier cap, applied on the percent scale -- but only when the
        # trophy is still below it: a trophy already above its cap has
        # outgrown the prior, and the data, not the cap, governs it.
        if last_rate < tier_cap(trophy):
            median = min(median, logit(tier_cap(trophy)))
        median = max(median, logit(0.01))
        medians.append(median)
        spread = math.sqrt(
            (WALK_SIGMA * math.sqrt(horizon_days)) ** 2
            + (0.45 * abs(delta)) ** 2
            + (0.25 * max(0.0, dx)) ** 2
            + min(resid, 0.6) ** 2
            + 0.12 ** 2
        )
        spreads.append(min(1.7, max(0.30, spread)))

    # Structure: the platinum needs every other trophy, so it cannot exceed
    # the lowest other median; story chapters cannot rise along the story.
    if medians:
        plat_idx = [i for i, t in enumerate(trophies)
                    if t.get("tier") == "platinum"]
        others = [medians[i] for i in range(len(trophies))
                  if i not in plat_idx]
        for i in plat_idx:
            if others:
                medians[i] = min(medians[i], min(others))
        chain = []
        for i, t in enumerate(trophies):
            desc = t.get("description") or ""
            m = re.search(r"Complete Chapter (\d+)", desc)
            if m:
                chain.append((int(m.group(1)), i))
            elif desc.startswith("Complete the final mission"):
                chain.append((99, i))
        chain.sort()
        running = None
        for _chapter, i in chain:
            if running is None or medians[i] < running:
                running = medians[i]
            else:
                medians[i] = running

    rows = []
    for idx, trophy in enumerate(trophies):
        qs = []
        prev = None
        for z in zvals:
            q = round4(logistic(medians[idx] + z * spreads[idx]))
            q = min(100.0, max(0.0, q))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
