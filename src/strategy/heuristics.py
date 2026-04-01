"""Aspiration-based fallback negotiator (no LLM needed)."""

from __future__ import annotations

from strategy.models import Observation, ProposalResponse, AcceptResponse, QUANTITIES


def compute_item_value_density(valuations: list[int], quantities: list[int]) -> list[tuple[int, float]]:
    """Returns list of (item_index, value_per_unit) sorted by density descending."""
    densities = []
    for i, (v, q) in enumerate(zip(valuations, quantities)):
        densities.append((i, v))  # value per unit
    densities.sort(key=lambda x: x[1], reverse=True)
    return densities


def greedy_allocation(
    valuations: list[int],
    quantities: list[int],
    target_value: float,
) -> list[int]:
    """Greedily allocate items to reach target_value, prioritizing highest-value items."""
    allocation = [0] * len(quantities)
    current_value = 0.0
    densities = compute_item_value_density(valuations, quantities)

    for item_idx, val_per_unit in densities:
        if current_value >= target_value:
            break
        needed = target_value - current_value
        units_needed = min(quantities[item_idx], max(1, int(needed / val_per_unit + 0.5)))
        allocation[item_idx] = units_needed
        current_value += units_needed * val_per_unit

    return allocation


def aspiration_propose(obs: Observation) -> ProposalResponse:
    """Aspiration-based proposal: target ~75% of total value, adapting by round."""
    total = obs.total_value
    # Start aspiring high, concede over rounds
    aspiration_level = max(0.6, 0.85 - 0.05 * (obs.round_index - 1))
    target = max(obs.batna_self, total * aspiration_level)

    allocation_self = greedy_allocation(obs.valuations_self, obs.quantities, target)

    # Ensure valid: cap at quantities
    allocation_self = [min(a, q) for a, q in zip(allocation_self, obs.quantities)]
    allocation_other = [q - a for q, a in zip(obs.quantities, allocation_self)]

    self_value = sum(v * a for v, a in zip(obs.valuations_self, allocation_self))
    return ProposalResponse(
        allocation_self=allocation_self,
        allocation_other=allocation_other,
        reason=f"Aspiration heuristic: targeting {aspiration_level:.0%} of total value ({self_value}/{total})",
    )


def aspiration_accept_or_reject(obs: Observation) -> AcceptResponse:
    """Accept if offer value exceeds BATNA adjusted for time pressure."""
    if obs.offer_value is None:
        # Try to compute from pending offer
        alloc = obs.pending_offer_allocation
        if alloc:
            offer_val = sum(v * a for v, a in zip(obs.valuations_self, alloc))
        else:
            return AcceptResponse(accept=False, reason="Cannot determine offer value")
    else:
        offer_val = obs.offer_value

    # Discount-adjusted threshold: accept if offer > BATNA
    # In later rounds, be more willing to accept
    rounds_left = obs.max_rounds - obs.round_index
    discount_pressure = obs.discount ** rounds_left
    threshold = obs.batna_self * discount_pressure

    # Also consider: accepting now vs counter-offering next round (discounted)
    accept = offer_val >= threshold or offer_val >= obs.batna_self

    return AcceptResponse(
        accept=accept,
        reason=f"Offer value {offer_val} vs threshold {threshold:.0f} (BATNA={obs.batna_self}, rounds_left={rounds_left})",
    )
