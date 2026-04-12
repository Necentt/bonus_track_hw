"""Aspiration-based fallback negotiator (no LLM needed).

Designed to be Pareto-efficient: we keep items we value most and give opponent
items we value least. On random valuations, this tends to also benefit opponent
(asymmetric preferences), boosting Utilitarian Welfare.
"""

from __future__ import annotations

from strategy.models import Observation, ProposalResponse, AcceptResponse, QUANTITIES


# Expected per-unit valuation for an unknown opponent (uniform[1, 99] mean).
EXPECTED_OPPONENT_VAL_PER_UNIT = 50


def pareto_propose(obs: Observation) -> ProposalResponse:
    """Pareto-efficient aspiration proposal.

    Strategy:
    1. Aspiration level decays over rounds: 0.78, 0.70, 0.62, ...
    2. Final round: aim at max(BATNA + small surplus, 0.55 * total).
    3. Keep items where self-valuation is HIGHEST per unit.
    4. Give items where self-valuation is LOWEST (opponent likely values them more, on average).
    5. Ensure opponent's allocation has enough EXPECTED value to be accepted
       (~ opponent's estimated BATNA + margin).
    """
    total = obs.total_value
    rounds_left = obs.max_rounds - obs.round_index

    # Aspiration schedule — slightly less greedy than before to push UW up
    if rounds_left == 0:
        aspiration = 0.55
    else:
        aspiration = max(0.58, 0.80 - 0.08 * (obs.round_index - 1))

    target_self_value = max(obs.batna_self, int(total * aspiration))

    # Sort items by our per-unit valuation descending — keep highest-value items first.
    item_order = sorted(
        range(len(obs.valuations_self)),
        key=lambda i: obs.valuations_self[i],
        reverse=True,
    )

    allocation_self = [0] * len(obs.quantities)
    current_value = 0

    # Greedy: take units of highest-value items until target reached.
    for item_idx in item_order:
        if current_value >= target_self_value:
            break
        val = obs.valuations_self[item_idx]
        qty = obs.quantities[item_idx]
        if val <= 0 or qty <= 0:
            continue
        # How many units do we need to reach target?
        remaining = target_self_value - current_value
        units = min(qty, max(1, (remaining + val - 1) // val))
        allocation_self[item_idx] = units
        current_value += units * val

    # Ensure we keep at least ONE unit of our highest-value item if possible
    # (even if target was already reached by accident).
    top_item = item_order[0]
    if allocation_self[top_item] == 0 and obs.quantities[top_item] > 0:
        allocation_self[top_item] = 1

    # Cap at quantities.
    allocation_self = [min(a, q) for a, q in zip(allocation_self, obs.quantities)]
    allocation_other = [q - a for q, a in zip(obs.quantities, allocation_self)]

    # Check opponent gets something substantial (Pareto sanity):
    # Expected opponent value on the items we give them.
    expected_opp_value = sum(
        EXPECTED_OPPONENT_VAL_PER_UNIT * a for a in allocation_other
    )
    # If opponent expected value is very low, give them an extra unit of our lowest-value item.
    if expected_opp_value < 150 and obs.round_index >= 2:
        for item_idx in reversed(item_order):
            if allocation_self[item_idx] > 0:
                allocation_self[item_idx] -= 1
                allocation_other[item_idx] += 1
                break

    self_value = sum(v * a for v, a in zip(obs.valuations_self, allocation_self))
    return ProposalResponse(
        allocation_self=allocation_self,
        allocation_other=allocation_other,
        reason=f"Pareto heuristic: kept high-value items, value={self_value}/{total} ({aspiration:.0%}), BATNA={obs.batna_self}",
    )


def smart_accept_or_reject(obs: Observation) -> AcceptResponse:
    """Accept if offer exceeds BATNA, especially in late rounds.

    Strategy:
    1. Compute the offer's value to us.
    2. In the FINAL round: accept any offer >= BATNA (rejecting means BATNA anyway).
    3. Otherwise: accept if offer_value >= max(BATNA, threshold based on rounds left).
    4. Always accept if offer is within ~5% of our likely next proposal.
    """
    # Compute offer value
    if obs.offer_value is not None:
        offer_val = obs.offer_value
    else:
        alloc = obs.pending_offer_allocation
        if alloc:
            offer_val = sum(v * a for v, a in zip(obs.valuations_self, alloc))
        else:
            return AcceptResponse(accept=False, reason="No offer details")

    rounds_left = obs.max_rounds - obs.round_index
    final_round = rounds_left == 0

    # Rule 1: final round — accept if at or above BATNA (no downside to accepting).
    if final_round:
        accept = offer_val >= obs.batna_self
        return AcceptResponse(
            accept=accept,
            reason=f"Final round: offer {offer_val} vs BATNA {obs.batna_self} -> {'ACCEPT' if accept else 'WALK (offer < BATNA)'}",
        )

    # Rule 2: clearly above BATNA — accept. Rejection cascades hurt UW/NWA too much.
    if offer_val >= 1.10 * obs.batna_self:
        return AcceptResponse(
            accept=True,
            reason=f"Offer {offer_val} >= 1.10 * BATNA {obs.batna_self}: accept to secure welfare.",
        )

    # Rule 3: clearly below BATNA — reject.
    if offer_val < 0.85 * obs.batna_self:
        return AcceptResponse(
            accept=False,
            reason=f"Offer {offer_val} < 0.85 * BATNA {obs.batna_self}: exploitative, reject.",
        )

    # Rule 4: borderline (between 0.85 and 1.10 BATNA).
    # Accept if limited rounds remain — can't recover from walk-away.
    if rounds_left <= 1:
        accept = offer_val >= obs.batna_self
        return AcceptResponse(
            accept=accept,
            reason=f"Borderline {offer_val} near BATNA {obs.batna_self}, {rounds_left} rounds left: {'ACCEPT' if accept else 'WALK'}",
        )

    # Rule 5: more rounds left, compare to EV of rejection.
    total = obs.total_value
    expected_next = 0.6 * total * obs.discount
    threshold = max(obs.batna_self, expected_next * 0.7)
    accept = offer_val >= threshold

    return AcceptResponse(
        accept=accept,
        reason=f"Offer {offer_val} vs threshold {threshold:.0f} (BATNA={obs.batna_self}, rounds_left={rounds_left})",
    )


# Backwards-compatible aliases (used by existing tests and nodes)
aspiration_propose = pareto_propose
aspiration_accept_or_reject = smart_accept_or_reject
