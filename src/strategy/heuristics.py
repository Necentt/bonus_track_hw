"""Deterministic Nash-welfare maximizing negotiator.

Strategy:
- Log-Nash greedy allocation maximizes √(u_self × u_opp) on integer lattice
- EF1 seed: give opponent at least 1 unit of each item type before greedy pass
- Opponent utility floor: ensure opponent gets >=25%→35% of their expected total
- Opponent valuation inference: inverse-of-own-values prior, updated from history
- Accept/Reject: hard BATNA floor + progress-tiered thresholds

No LLM in the main path → deterministic → low MENE Regret.
"""

from __future__ import annotations

import math
from strategy.models import Observation, ProposalResponse, AcceptResponse


# Uniform[1, 99] mean.
EXPECTED_OPPONENT_VAL_PER_UNIT = 50
# Prior assumption about opponent total value when unknown.
DEFAULT_OPPONENT_MAX_VALUE = 50 * 12  # 50 mean × sum([7,4,1]) = 600


def _estimate_opponent_values(
    obs: Observation,
    history: list[dict] | None = None,
) -> list[float]:
    """Estimate opponent per-unit valuations.

    Prior: inverse of our own valuations (they probably value what we don't).
    Normalize to have same sum as our own valuations so scales match.
    If history exists, we can refine by watching what opp KEPT in their proposals.
    """
    own = obs.valuations_self
    max_own = max(own) if own else 99
    # Inverse prior: items we value least, they value most.
    prior = [float(max_own + 1 - v) for v in own]
    # Normalize to sum match.
    target_sum = sum(own)
    prior_sum = sum(prior) or 1.0
    prior = [p * target_sum / prior_sum for p in prior]

    # Update from history: weight each item by fraction opponent kept in their proposals.
    if history:
        kept_fractions = [0.0] * len(own)
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
                    kept_fractions[i] += (k / q) * w
                    weights[i] += w
            recency += 1
        for i in range(len(own)):
            if weights[i] > 0:
                avg_kept = kept_fractions[i] / weights[i]
                # Map avg_kept ∈ [0, 1] → multiplier ∈ [0.5, 2.0].
                mult = 0.5 + 1.5 * avg_kept
                prior[i] *= mult

    return prior


def _log_nash_greedy(
    own_vals: list[int],
    opp_vals: list[float],
    quantities: list[int],
    target_self_min: float,
    target_opp_min: float,
) -> tuple[list[int], list[int]]:
    """Greedy allocation maximizing log(u_self) + log(u_opp) = log(u_self * u_opp).

    Start by seeding opponent with 1 unit of each item (EF1 insurance), then
    allocate each remaining unit to whoever gains more log-utility.
    """
    n = len(quantities)
    # EF1 seed: give opponent 1 unit of each item where possible.
    alloc_self = [0] * n
    alloc_other = [min(1, q) for q in quantities]
    # The rest — greedy by log-utility gain.
    remaining = [q - alloc_other[i] for i, q in enumerate(quantities)]

    # Track running totals.
    total_self = 1.0  # ε to avoid log(0)
    total_opp = sum(ov * ao for ov, ao in zip(opp_vals, alloc_other)) + 1.0

    # Total units to distribute.
    total_units = sum(remaining)
    for _ in range(total_units):
        best_idx = -1
        best_to_self = False
        best_gain = -math.inf
        for i in range(n):
            if remaining[i] <= 0:
                continue
            sv = own_vals[i]
            ov = opp_vals[i]
            # Gain if given to SELF: Δlog(total_self) = log((total_self+sv)/total_self).
            gain_self = math.log((total_self + sv) / total_self) if sv > 0 else 0.0
            gain_opp = math.log((total_opp + ov) / total_opp) if ov > 0 else 0.0
            if gain_self > best_gain:
                best_gain = gain_self
                best_idx = i
                best_to_self = True
            if gain_opp > best_gain:
                best_gain = gain_opp
                best_idx = i
                best_to_self = False
        if best_idx < 0:
            break
        if best_to_self:
            alloc_self[best_idx] += 1
            total_self += own_vals[best_idx]
        else:
            alloc_other[best_idx] += 1
            total_opp += opp_vals[best_idx]
        remaining[best_idx] -= 1

    # Self-floor: if below target_self_min, steal units by best self_val/opp_val ratio.
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
        if total_self >= target_self_min:
            break

    # Opponent-floor: if below target_opp_min, give back by best opp_val/self_val ratio.
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
        if total_opp >= target_opp_min:
            break

    return alloc_self, alloc_other


def aspiration_propose(
    obs: Observation,
    history: list[dict] | None = None,
) -> ProposalResponse:
    """Nash-welfare greedy proposal with opponent-floor and self-floor."""
    total = obs.total_value
    rounds_left = obs.max_rounds - obs.round_index
    progress = obs.round_index / max(1, obs.max_rounds)

    # Self-fraction target: 0.60 early → 0.50 late.
    self_frac = max(0.50, 0.60 - 0.10 * progress)
    # Opponent-fraction target: 0.25 early → 0.35 late.
    opp_frac = min(0.35, 0.25 + 0.10 * progress)

    # Final-round override: don't insist on self target, just exceed BATNA.
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
    )

    # Sanity: non-negative, sums correct.
    alloc_self = [max(0, min(a, q)) for a, q in zip(alloc_self, obs.quantities)]
    alloc_other = [q - a for q, a in zip(obs.quantities, alloc_self)]

    self_value = sum(v * a for v, a in zip(obs.valuations_self, alloc_self))
    return ProposalResponse(
        allocation_self=alloc_self,
        allocation_other=alloc_other,
        reason=f"NashGreedy r{obs.round_index}/{obs.max_rounds} u_self={self_value}/{total} batna={obs.batna_self}",
    )


def aspiration_accept_or_reject(
    obs: Observation,
    history: list[dict] | None = None,
) -> AcceptResponse:
    """Progress-tiered accept with hard BATNA floor."""
    if obs.offer_value is not None:
        offer_val = obs.offer_value
    else:
        alloc = obs.pending_offer_allocation
        if alloc:
            offer_val = sum(v * a for v, a in zip(obs.valuations_self, alloc))
        else:
            return AcceptResponse(accept=False, reason="no offer details")

    rounds_left = obs.max_rounds - obs.round_index
    progress = obs.round_index / max(1, obs.max_rounds)

    # Final round: accept iff >= BATNA.
    if rounds_left <= 0:
        accept = offer_val >= obs.batna_self
        return AcceptResponse(
            accept=accept,
            reason=f"final: offer={offer_val} batna={obs.batna_self}",
        )

    # Hard floor: never accept below BATNA.
    if offer_val < obs.batna_self:
        return AcceptResponse(
            accept=False,
            reason=f"below BATNA: offer={offer_val} batna={obs.batna_self}",
        )

    # Late-game (progress >= 0.60): accept anything >= 1.02 × BATNA.
    if progress >= 0.60 and offer_val >= 1.02 * obs.batna_self:
        return AcceptResponse(
            accept=True,
            reason=f"late-game accept: offer={offer_val} batna={obs.batna_self}",
        )

    # Progress-tiered threshold against discounted expected counter.
    total = obs.total_value
    expected_counter = 0.55 * total  # what we'd propose next round
    discounted = expected_counter * obs.discount

    if progress < 0.33:
        threshold_mult = 0.90
    elif progress < 0.67:
        threshold_mult = 0.80
    else:
        threshold_mult = 0.60

    threshold = max(float(obs.batna_self), discounted * threshold_mult)
    accept = offer_val >= threshold
    return AcceptResponse(
        accept=accept,
        reason=f"offer={offer_val} threshold={threshold:.0f} progress={progress:.2f}",
    )


pareto_propose = aspiration_propose
smart_accept_or_reject = aspiration_accept_or_reject
