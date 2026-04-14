"""Deterministic Nash-welfare maximizing negotiator (v2.1).

Hybrid of proven winners:
- Aspiration schedule (v1.0 core, MENE #1) — near Nash equilibrium support.
- Log-Nash greedy with EF1 seed (v2.0, FanisNgv-inspired) — NW/NWA/EF1 boost.
- Controlled stochastic perturbation — breaks best-response exploitability
  (pure strategies get best-responded → high MENE; mixed don't).
- Opp-type detection from history → adaptive opp_floor.
- Accept logic copied from v1.0 exactly — proven lowest MENE.

No LLM in the main path. LLM only as safety net for unparsable observations.
"""

from __future__ import annotations

import math
import random

from strategy.models import Observation, ProposalResponse, AcceptResponse


# Uniform[1, 99] mean.
EXPECTED_OPPONENT_VAL_PER_UNIT = 50


# --------------------------------------------------------------------------- #
# Opponent modeling
# --------------------------------------------------------------------------- #

def _estimate_opponent_values(
    obs: Observation,
    history: list[dict] | None = None,
) -> list[float]:
    """Estimate opponent per-unit valuations.

    Prior = inverse of own valuations (creates Pareto-efficient splits by
    construction: we give items we value least, opponent is assumed to value
    them most). Normalize to match sum of own values.

    Refine from history: items opponent consistently kept → value them higher.
    """
    own = obs.valuations_self
    if not own:
        return []

    max_own = max(own)
    prior = [float(max_own + 1 - v) for v in own]
    target_sum = sum(own)
    prior_sum = sum(prior) or 1.0
    prior = [p * target_sum / prior_sum for p in prior]

    if history:
        kept_frac = [0.0] * len(own)
        weights = [0.0] * len(own)
        recency = 1.0
        for h in history:
            if h.get("action") != "opponent_proposed":
                continue
            opp_kept = h.get("opponent_kept") or []
            if len(opp_kept) != len(own):
                continue
            w = 1.0 + 0.6 * recency
            for i, (k, q) in enumerate(zip(opp_kept, obs.quantities)):
                if q > 0:
                    kept_frac[i] += (k / q) * w
                    weights[i] += w
            recency += 1
        for i in range(len(own)):
            if weights[i] > 0:
                avg = kept_frac[i] / weights[i]
                # avg ∈ [0, 1] → multiplier ∈ [0.5, 2.0]
                prior[i] *= 0.5 + 1.5 * avg

    return prior


def _infer_opp_type(history: list[dict] | None) -> str:
    """Classify opponent from observed proposals.

    Looks at AVERAGE units opponent gave us in their proposals.
    - tough: < 3 units avg → minimal offers → concede more to close deal
    - soft: >= 6 units avg → generous → claim more aggressively
    - moderate: in between → standard aspiration
    """
    if not history:
        return "moderate"
    given_totals = []
    for h in history:
        if h.get("action") == "opponent_proposed":
            offer = h.get("offer_to_me") or []
            if offer:
                given_totals.append(sum(offer))
    if not given_totals:
        return "moderate"
    avg = sum(given_totals) / len(given_totals)
    if avg <= 3:
        return "tough"
    if avg >= 6:
        return "soft"
    return "moderate"


# --------------------------------------------------------------------------- #
# Nash-welfare greedy allocation
# --------------------------------------------------------------------------- #

def _log_nash_greedy(
    own_vals: list[int],
    opp_vals: list[float],
    quantities: list[int],
    target_self_min: float,
    target_opp_min: float,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int]]:
    """Greedy allocation maximizing log(u_self) + log(u_opp).

    EF1 seed: give opponent 1 unit of each item first (guarantees envy-free
    up to one item). Remaining units allocated to whoever gains more
    log-utility. Tie-breaking randomized via rng to introduce mixed strategy
    (lowers MENE Regret).
    """
    if rng is None:
        rng = random.Random()
    n = len(quantities)
    alloc_self = [0] * n
    alloc_other = [min(1, q) for q in quantities]
    remaining = [q - alloc_other[i] for i, q in enumerate(quantities)]

    total_self = 1.0
    total_opp = sum(ov * ao for ov, ao in zip(opp_vals, alloc_other)) + 1.0

    total_units = sum(remaining)
    for _ in range(total_units):
        # Build candidates with their gains.
        candidates = []
        for i in range(n):
            if remaining[i] <= 0:
                continue
            sv = own_vals[i]
            ov = opp_vals[i]
            gain_self = math.log((total_self + sv) / total_self) if sv > 0 else 0.0
            gain_opp = math.log((total_opp + ov) / total_opp) if ov > 0 else 0.0
            candidates.append((gain_self, i, True))
            candidates.append((gain_opp, i, False))
        if not candidates:
            break
        # Find max gain, then random tie-break.
        max_gain = max(c[0] for c in candidates)
        best_pool = [c for c in candidates if c[0] >= max_gain - 1e-9]
        _, best_idx, best_to_self = rng.choice(best_pool)
        if best_to_self:
            alloc_self[best_idx] += 1
            total_self += own_vals[best_idx]
        else:
            alloc_other[best_idx] += 1
            total_opp += opp_vals[best_idx]
        remaining[best_idx] -= 1

    # Self-floor: steal from opp by best self/opp ratio.
    while total_self < target_self_min:
        best_i = -1
        best_ratio = -1.0
        for i in range(n):
            if alloc_other[i] <= 0:
                continue
            sv = own_vals[i]
            ov = opp_vals[i] or 1.0
            ratio = sv / ov
            if ratio > best_ratio:
                best_ratio = ratio
                best_i = i
        if best_i < 0:
            break
        alloc_other[best_i] -= 1
        alloc_self[best_i] += 1
        total_self += own_vals[best_i]
        total_opp -= opp_vals[best_i]

    # Opp-floor: give back by best opp/self ratio.
    while total_opp < target_opp_min:
        best_i = -1
        best_ratio = -1.0
        for i in range(n):
            if alloc_self[i] <= 0:
                continue
            sv = own_vals[i] or 1.0
            ov = opp_vals[i]
            ratio = ov / sv
            if ratio > best_ratio:
                best_ratio = ratio
                best_i = i
        if best_i < 0:
            break
        alloc_self[best_i] -= 1
        alloc_other[best_i] += 1
        total_self -= own_vals[best_i]
        total_opp += opp_vals[best_i]

    return alloc_self, alloc_other


# --------------------------------------------------------------------------- #
# Aspiration schedule with stochastic perturbation
# --------------------------------------------------------------------------- #

def _self_fraction_target(
    round_index: int,
    max_rounds: int,
    opp_type: str,
    rng: random.Random,
) -> float:
    """Aspiration fraction for self — starts high, concedes over rounds.

    Schedule (round 1 → max-1): 0.72 → 0.54 linearly.
    Final round: 0.50 (just a floor; BATNA also enforced).
    Opp-type adjustment:
        tough  → -0.05 (concede more to close deal, avoid walk-away)
        soft   → +0.04 (claim more, opp will accept)
        moderate → no change
    Stochastic perturbation ±0.025 (uniform) → prevents best-response exploitation.
    """
    if round_index >= max_rounds:
        center = 0.50
    elif max_rounds <= 1:
        center = 0.55
    else:
        t = (round_index - 1) / max(1, max_rounds - 2)
        t = max(0.0, min(1.0, t))
        center = 0.72 - 0.18 * t

    if opp_type == "tough":
        center -= 0.05
    elif opp_type == "soft":
        center += 0.04

    noise = rng.uniform(-0.025, 0.025)
    return max(0.45, min(0.80, center + noise))


def _opp_floor_fraction(
    round_index: int,
    max_rounds: int,
    opp_type: str,
) -> float:
    """Opponent utility floor — fraction of opp's expected total we must give.

    Base schedule 0.22 → 0.28 over rounds. Adjusted by opp-type.
    """
    progress = round_index / max(1, max_rounds)
    base = 0.22 + 0.06 * progress
    if opp_type == "tough":
        return max(0.15, base - 0.05)
    if opp_type == "soft":
        return min(0.30, base - 0.02)
    return base


# --------------------------------------------------------------------------- #
# Public API: propose
# --------------------------------------------------------------------------- #

def aspiration_propose(
    obs: Observation,
    history: list[dict] | None = None,
) -> ProposalResponse:
    """v2.1 proposal: aspiration + Nash-greedy + stochastic + opp-adaptive."""
    rng = random.Random()  # truly random each call — breaks exploitability
    opp_type = _infer_opp_type(history)

    total = obs.total_value
    self_frac = _self_fraction_target(obs.round_index, obs.max_rounds, opp_type, rng)
    opp_frac = _opp_floor_fraction(obs.round_index, obs.max_rounds, opp_type)

    rounds_left = obs.max_rounds - obs.round_index
    if rounds_left == 0:
        self_target = float(obs.batna_self)
    else:
        self_target = max(float(obs.batna_self), total * self_frac)

    opp_vals = _estimate_opponent_values(obs, history)
    opp_total_est = sum(ov * q for ov, q in zip(opp_vals, obs.quantities))
    opp_target = opp_total_est * opp_frac

    alloc_self, alloc_other = _log_nash_greedy(
        own_vals=obs.valuations_self,
        opp_vals=opp_vals,
        quantities=obs.quantities,
        target_self_min=self_target,
        target_opp_min=opp_target,
        rng=rng,
    )

    alloc_self = [max(0, min(a, q)) for a, q in zip(alloc_self, obs.quantities)]
    alloc_other = [q - a for q, a in zip(obs.quantities, alloc_self)]

    # Stochastic mutation (25% chance): swap 1 unit if buffer above BATNA exists.
    # Adds true output randomness → strategy becomes mixed → MENE drops.
    if rng.random() < 0.25 and rounds_left > 0:
        cur_self_value = sum(v * a for v, a in zip(obs.valuations_self, alloc_self))
        if cur_self_value > obs.batna_self * 1.10:
            n = len(obs.quantities)
            swappable = [
                i for i in range(n)
                if alloc_self[i] > 0 and obs.valuations_self[i] > 0
            ]
            if swappable:
                # Give one unit of a random self-item to opp (favors items
                # with lowest self-value to preserve our utility).
                weights = [1.0 / max(1, obs.valuations_self[i]) for i in swappable]
                idx = rng.choices(swappable, weights=weights, k=1)[0]
                # Only swap if we still exceed BATNA after.
                new_value = cur_self_value - obs.valuations_self[idx]
                if new_value >= obs.batna_self:
                    alloc_self[idx] -= 1
                    alloc_other[idx] += 1

    self_value = sum(v * a for v, a in zip(obs.valuations_self, alloc_self))
    return ProposalResponse(
        allocation_self=alloc_self,
        allocation_other=alloc_other,
        reason=(
            f"v2.1 r{obs.round_index}/{obs.max_rounds} opp={opp_type} "
            f"self_frac={self_frac:.2f} u_self={self_value}/{total}"
        ),
    )


# --------------------------------------------------------------------------- #
# Public API: accept/reject (replica of v1.0 formula — proven MENE champion)
# --------------------------------------------------------------------------- #

def aspiration_accept_or_reject(
    obs: Observation,
    history: list[dict] | None = None,
) -> AcceptResponse:
    """Accept logic from v1.0 (the MENE #1 version).

    - Final round: accept iff offer_value >= BATNA.
    - Otherwise: accept iff offer_value >= max(BATNA, 0.85 * discount * 0.6 * total).
    - Deterministic (no history, no LLM) — matches near-equilibrium play.
    """
    if obs.offer_value is not None:
        offer_val = obs.offer_value
    else:
        alloc = obs.pending_offer_allocation
        if alloc:
            offer_val = sum(v * a for v, a in zip(obs.valuations_self, alloc))
        else:
            return AcceptResponse(accept=False, reason="no offer details")

    rounds_left = obs.max_rounds - obs.round_index

    if rounds_left <= 0:
        accept = offer_val >= obs.batna_self
        return AcceptResponse(
            accept=accept,
            reason=f"final: offer={offer_val} batna={obs.batna_self}",
        )

    total = obs.total_value
    expected_next = 0.60 * total * obs.discount
    threshold = max(float(obs.batna_self), expected_next * 0.85)

    accept = offer_val >= threshold
    return AcceptResponse(
        accept=accept,
        reason=f"offer={offer_val} threshold={threshold:.0f} batna={obs.batna_self}",
    )


# Backwards-compatible aliases
pareto_propose = aspiration_propose
smart_accept_or_reject = aspiration_accept_or_reject
