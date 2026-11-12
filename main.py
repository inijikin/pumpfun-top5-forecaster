#!/usr/bin/env python3
"""Pump.fun top-5 forecaster for make.fun competition pumpfun-top5-forecast-01.

Reads the house's 10-minute snapshots (/input/history.ndjson + task.json) and writes
one forecast (/output/forecast.json): per-place probabilities for each coin, the top 5
in order, and per-place market-cap size-bin probabilities.

Method: Monte Carlo over each coin's log market-cap trajectory, simulated to each of
the seven answer reads and averaged (the one forecast is scored at every read).

Each coin's log-cap is a jump-diffusion with mean reversion:
  - diffusion: recent per-step drift (winsorized, so a single pump does not read
    as a trend, and shrunk toward the cross-sectional prior), saturating over a
    ~3h mean-reversion time; volatility from its own recent increments, shrunk
    toward the prior of the coins that actually move (top-50 coins run ~0.12 log
    per step; the median coin barely moves at all);
  - jumps: pump.fun coins sit flat then jump. A per-hour hazard adds a positive
    jump or a drop (mean 0.30 log) at a random time before the read, its effect
    fading over ~6h, so rockets matter at short horizons and fade at long ones;
  - death: a per-hour hazard drops a coin's cap (most coins fade within hours).
A per-place newcomer reserve is left unnamed and grows with the horizon: coins
born after the history ends take a growing share of the top 5 (in the practice
window 2 of the final top 5 were launched after the cut), and no history can name
them, so part of every place must stay open for them.
Sizes mix the simulated distribution with a broad normal around the place's
current cap plus growth (the top of the chart rises as fresh rockets arrive),
so a place's size is never bet on a narrow spike.

Standard library only, no network, deterministic: a fixed seed and fully ordered
iteration, so two runs on the same history write byte-identical output.
"""

import json
import math
import os
import random
from datetime import datetime, timezone


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


SEED = 20261112
N_SIM = int(os.environ.get("MF_N_SIM", "8000"))
UNIVERSE = 200            # coins simulated (top by last cap + recent top-20)
TAU_STEPS = 18.0          # mean-reversion time in 10-minute steps (~3h)
DRIFT_SHRINK_K = 3.0      # pseudo-count shrinking a coin's drift to the prior
VOL_SHRINK_K = 3.0
VOL_FLOOR = 0.08          # per-step log10 sd
VOL_CAP = 0.40
DRIFT_CLAMP = 0.12        # per-step log10 drift clamp
WINSOR = 0.12             # increments clipped here for drift/vol estimation
DEATH_LAMBDA_PER_H = _f("MF_DEATH", 0.07)  # hazard of a collapse per hour
DEATH_DROP = 1.5          # log10 drop when a coin collapses
YOUNG_VOL_MULT = 1.3      # coins < 1h old are wilder
POOL_VOL_MULT = 0.85      # graduated (pool) coins are steadier
JUMP_UP_LAMBDA_PER_H = _f("MF_JUMP_UP", 0.08)    # hazard of a pump jump per hour
JUMP_DOWN_LAMBDA_PER_H = _f("MF_JUMP_DOWN", 0.12)  # hazard of a drop per hour
JUMP_MEAN = 0.30          # mean log10 jump/drop size
JUMP_SD = 0.15
JUMP_DECAY_STEPS = 36.0   # a jump's effect fades over ~6h
JUMP_YOUNG_MULT = 1.5     # young coins jump more
JUMP_POOL_MULT = 0.7      # graduated coins jump less
# Newcomer reserve per place, growing with horizon h (hours since the cut):
#   q(h) = Q0 + (QMAX - Q0) * (1 - exp(-h / TAU_NEW_H))
NEWCOMER_Q0 = _f("MF_Q0", 0.30)
NEWCOMER_QMAX = _f("MF_QMAX", 0.65)
NEWCOMER_TAU_H = 2.5
SIZE_MC_WEIGHT = 0.6      # weight of the simulated size distribution
SIZE_NORMAL_SD = _f("MF_SIZE_SD", 0.35)     # sd of the broad-normal size component
SIZE_GROWTH_MAX = _f("MF_SIZE_GROWTH", 0.4)  # how far the top of the chart rises, in log10
SIZE_GROWTH_TAU_STEPS = 12.0
SIZE_UNNAMED = 0.10       # fraction of each place's size mass left unnamed
MIN_P = 0.001             # smallest probability row emitted


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def size_of(cap_lamports):
    """log10 of a market cap in SOL, as the scorer takes it."""
    return math.log10(int(cap_lamports)) - 9.0


def bin_of(size, edges):
    """Size bin 0..81 for a log10 cap in SOL, per the rules."""
    if size < edges[0]:
        return 0
    if size >= edges[-1]:
        return 81
    lo, hi = 0, len(edges) - 1
    while lo < hi:  # first edge > size
        mid = (lo + hi) // 2
        if edges[mid] <= size:
            lo = mid + 1
        else:
            hi = mid
    return lo  # edges[lo-1] <= size < edges[lo]


def normal_bin_probs(mean, sd, edges):
    """P(bin) for a normal in log10 cap, over the 82 bins."""
    out = {}

    def cdf(x):
        return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))

    prev = cdf(edges[0])
    out[0] = prev
    for i in range(1, len(edges)):
        cur = cdf(edges[i])
        out[i] = cur - prev
        prev = cur
    out[81] = 1.0 - prev
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(in_dir, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    reads = [parse_time(r) for r in task["reads"]]
    step_min = float(task.get("cadence_minutes", 10))

    marks = []
    with open(os.path.join(in_dir, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark")
    marks.sort(key=lambda m: m["at"])
    last_at = parse_time(marks[-1]["at"])
    n_marks = len(marks)

    # Per-coin series: mint -> {step_idx: (logcap, venue, complete, created_at, rank)}
    series = {}
    for i, m in enumerate(marks):
        for r in m["rows"]:
            mint = r["mint"]
            entry = series.setdefault(mint, {})
            entry[i] = (
                size_of(r["market_cap_lamports"]),
                r.get("venue", "curve"),
                bool(r.get("complete", False)),
                r.get("created_at"),
                int(r.get("rank", 999)),
            )
    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    if len(last_rows) < places:
        raise SystemExit("the last mark ranks fewer coins than the places")

    # --- Per-coin drift / volatility from recent winsorized increments -----
    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    drifts_obs, vols_obs, top_vols = [], [], []
    per_coin = {}
    for mint, obs in series.items():
        idxs = sorted(obs)
        incs = []  # (age_steps_from_last, per-step increment)
        for a, b in zip(idxs, idxs[1:]):
            gap = b - a
            if 1 <= gap <= 3:
                incs.append((n_marks - 1 - b, (obs[b][0] - obs[a][0]) / gap))
        if not incs:
            continue
        clipped = [(age, min(max(d, -WINSOR), WINSOR)) for age, d in incs]
        wsum = sum(0.7 ** age for age, _ in clipped)
        drift = sum((0.7 ** age) * d for age, d in clipped) / wsum
        var = sum((0.7 ** age) * (d - drift) ** 2 for age, d in clipped) / wsum
        vol = math.sqrt(var) if var > 0 else 0.0
        drifts_obs.append(drift)
        vols_obs.append(vol)
        per_coin[mint] = (drift, vol, wsum)
        if min(obs[i][4] for i in idxs) <= 50:
            top_vols.append(vol)

    prior_drift = median(drifts_obs) if drifts_obs else -0.01
    # The prior volatility is the movers' volatility, not the median coin's:
    # top-50 coins run ~0.12 log per step; the median coin barely moves.
    prior_vol = median(top_vols) if top_vols else 0.12
    prior_vol = min(max(prior_vol, VOL_FLOOR), VOL_CAP)

    # --- Universe: top by last cap + any recent top-20 coin ----------------
    last_cap = {r["mint"]: size_of(r["market_cap_lamports"]) for r in last_rows}
    recent_top = set()
    for m in marks[-6:]:
        for r in sorted(m["rows"], key=lambda r: r["rank"])[:20]:
            recent_top.add(r["mint"])
    universe = sorted(last_cap, key=lambda mint: (-last_cap[mint], mint))[:UNIVERSE]
    for mint in sorted(recent_top):
        if mint not in universe and mint in last_cap:
            universe.append(mint)
    universe = universe[:250]

    # Per-coin forecast parameters.
    params = {}
    for mint in universe:
        x0 = last_cap[mint]
        obs = series[mint]
        last_i = max(obs)
        venue = obs[last_i][1]
        created = obs[last_i][3]
        age_h = None
        if created:
            try:
                age_h = (last_at - parse_time(created)).total_seconds() / 3600.0
            except ValueError:
                age_h = None
        if mint in per_coin:
            drift_o, vol_o, wsum = per_coin[mint]
            drift = (wsum * drift_o + DRIFT_SHRINK_K * prior_drift) / (wsum + DRIFT_SHRINK_K)
            vol = (wsum * vol_o + VOL_SHRINK_K * prior_vol) / (wsum + VOL_SHRINK_K)
        else:
            drift, vol = prior_drift, prior_vol
        drift = min(max(drift, -DRIFT_CLAMP), DRIFT_CLAMP)
        vol = min(max(vol, VOL_FLOOR), VOL_CAP)
        jump_mult = 1.0
        if age_h is not None and age_h < 1.0:
            vol *= YOUNG_VOL_MULT
            jump_mult *= JUMP_YOUNG_MULT
        if venue == "pool":
            vol *= POOL_VOL_MULT
            jump_mult *= JUMP_POOL_MULT
        params[mint] = (x0, drift, vol, jump_mult)

    # Horizons in 10-minute steps from the last mark to each answer read.
    horizons = []
    for r in reads:
        h = (r - last_at).total_seconds() / 60.0 / step_min
        horizons.append(max(h, 0.0))

    # --- Monte Carlo, per read (newcomer reserve grows with horizon) -------
    rng = random.Random(SEED)
    # prob_sum[place][mint], size_sum[place][bin]: averaged over reads
    prob_sum = {k: {} for k in range(1, places + 1)}
    size_sum = {k: {} for k in range(1, places + 1)}
    last_place_cap = {}
    for k in range(1, places + 1):
        last_place_cap[k] = size_of(last_rows[k - 1]["market_cap_lamports"])
    for h in horizons:
        h_hours = h * step_min / 60.0
        q = NEWCOMER_Q0 + (NEWCOMER_QMAX - NEWCOMER_Q0) * (1.0 - math.exp(-h_hours / NEWCOMER_TAU_H))
        growth = SIZE_GROWTH_MAX * (1.0 - math.exp(-h / SIZE_GROWTH_TAU_STEPS))
        decay = math.exp(-h / TAU_STEPS)
        decay2 = math.exp(-2.0 * h / TAU_STEPS)
        mu, sd, p_death, p_up, p_down = {}, {}, {}, {}, {}
        for mint, (x0, drift, vol, jump_mult) in params.items():
            mu[mint] = x0 + drift * TAU_STEPS * (1.0 - decay)
            var = vol * vol * (TAU_STEPS / 2.0) * (1.0 - decay2)
            sd[mint] = math.sqrt(var) if var > 0 else 0.0
            p_death[mint] = 1.0 - math.exp(-DEATH_LAMBDA_PER_H * h_hours)
            p_up[mint] = 1.0 - math.exp(-JUMP_UP_LAMBDA_PER_H * jump_mult * h_hours)
            p_down[mint] = 1.0 - math.exp(-JUMP_DOWN_LAMBDA_PER_H * jump_mult * h_hours)
        place_counts = {k: {} for k in range(1, places + 1)}
        size_counts = {k: {} for k in range(1, places + 1)}
        for _ in range(N_SIM):
            drawn = []
            for mint in universe:
                s = sd[mint]
                x = rng.gauss(mu[mint], s) if s > 0 else mu[mint]
                u = rng.random()
                if u < p_up[mint]:
                    t_jump = rng.random() * h if h > 0 else 0.0
                    fade = math.exp(-(h - t_jump) / JUMP_DECAY_STEPS) if h > 0 else 1.0
                    x += rng.gauss(JUMP_MEAN, JUMP_SD) * fade
                if rng.random() < p_down[mint]:
                    t_jump = rng.random() * h if h > 0 else 0.0
                    fade = math.exp(-(h - t_jump) / JUMP_DECAY_STEPS) if h > 0 else 1.0
                    x -= rng.gauss(JUMP_MEAN, JUMP_SD) * fade
                if rng.random() < p_death[mint]:
                    x -= DEATH_DROP
                drawn.append((x, mint))
            drawn.sort(key=lambda t: (-t[0], t[1]))
            for k in range(places):
                x, mint = drawn[k]
                place_counts[k + 1][mint] = place_counts[k + 1].get(mint, 0) + 1
                b = bin_of(x, edges)
                size_counts[k + 1][b] = size_counts[k + 1].get(b, 0) + 1
        for k in range(1, places + 1):
            for mint, c in place_counts[k].items():
                prob_sum[k][mint] = prob_sum[k].get(mint, 0.0) + (c / N_SIM) * (1.0 - q)
            normal = normal_bin_probs(last_place_cap[k] + growth, SIZE_NORMAL_SD, edges)
            for b in range(82):
                mc_p = size_counts[k].get(b, 0) / N_SIM
                mixed = SIZE_MC_WEIGHT * mc_p + (1.0 - SIZE_MC_WEIGHT) * normal.get(b, 0.0)
                size_sum[k][b] = size_sum[k].get(b, 0.0) + mixed * (1.0 - SIZE_UNNAMED)

    n_reads = len(horizons)
    probabilities = []
    for k in range(1, places + 1):
        for mint, p in sorted(prob_sum[k].items(), key=lambda kv: (-kv[1], kv[0])):
            p /= n_reads
            if p >= MIN_P:
                probabilities.append({"mint": mint, "place": k, "p": round(p, 6)})
    sizes = []
    for k in range(1, places + 1):
        for b, p in sorted(size_sum[k].items()):
            p /= n_reads
            if p >= MIN_P:
                sizes.append({"place": k, "bin": b, "p": round(p, 6)})

    # top5: greedy best coin per place, distinct.
    top5 = []
    used = set()
    for k in range(1, places + 1):
        best = None
        for row in probabilities:
            if row["place"] == k and row["mint"] not in used:
                best = row["mint"]
                break
        if best is None:  # fall back to the last mark's order
            for r in last_rows:
                if r["mint"] not in used:
                    best = r["mint"]
                    break
        top5.append(best)
        used.add(best)

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, sort_keys=True, separators=(",", ":")))
    print("wrote %s (%d probability rows, %d size rows)" % (out_path, len(probabilities), len(sizes)))
    print("top5:", top5)


if __name__ == "__main__":
    main()
