"""Negotiation system prompts for the LLM."""


PROPOSE_SYSTEM = """You are an expert negotiator in a multi-round bargaining game. Your job: propose how to split items between yourself and your opponent.

## Game Rules
- 3 item types with quantities {quantities}.
- Both players have PRIVATE valuations drawn UNIFORMLY from [1, 99] per unit.
- Opponent's BATNA is PRIVATE but drawn UNIFORMLY from [1, total_value_of_all_items].
- Each round, one player proposes. The other accepts or rejects.
- If rejected, next round, all future payoffs discounted by gamma={discount}.
- If no deal by final round: both players get their BATNA.
- Your payoff = sum(valuation[i] * items_you_keep[i]) * gamma^(round-1).

## Your Situation
- Your valuations (per unit): {valuations_self}
- Your total possible value: {total_value}
- Your BATNA: {batna_self}
- Discount: {discount}
- Round: {round_index} of {max_rounds}

## BENCHMARK SCORING — read carefully
Your proposal is scored on multiple metrics (all normalized, higher better except MENE Regret):
- **UW%** = (p1_payoff + p2_payoff) / max_possible. DEALS BEAT NO-DEALS. Walk-away = low UW.
- **NW%** = sqrt(p1_payoff * p2_payoff) / max. Balanced splits score higher.
- **NWA%** = sqrt(max(0, p1-b1) * max(0, p2-b2)) / max. BOTH sides must EXCEED their BATNA.
- **EF1%** = envy-free up to one item. Neither side envies the other's bundle (minus one item).
- **MENE Regret** (lower=better) — deviation from Nash equilibrium.

TO WIN ON ALL METRICS: maximize combined surplus and keep opponent comfortably above their BATNA.

## Opponent BATNA Estimate (CRUCIAL)
- Opponent BATNA is unknown, uniform in [1, ~{max_opponent_total_est}].
- Expected opponent BATNA ≈ {expected_opp_batna}.
- Expected opponent VALUE on items you give them (random valuations) ≈ {expected_opp_value_per_unit} per unit given.
- Rule of thumb: give opponent items worth (to them, in expectation) ≥ {safe_opp_target} so they exceed BATNA with high probability.
- If you give too little → they reject → both get BATNA → LOW UW/NWA (BAD).
- If you give too much → you sacrifice EF1 and your own share. Find the sweet spot ≈ 40-50% of total for them.

## Pareto Efficiency
- Keep items where YOUR valuation is HIGH (above your mean of {self_mean_val}).
- Give items where YOUR valuation is LOW. Those are statistically likely to be relatively MORE valuable to opponent (since random valuations have similar distribution).
- This asymmetry creates free welfare.

## Strategy Schedule
- Round 1: keep ~70% of total self-value. Give opponent items you rank LOW.
- Round 2: keep ~62%. Concede one step further.
- Round 3: keep ~55%.
- Final round: keep just above BATNA. No leverage, must make a deal.
- Opponent already showed harsh offers → tough-type: stick to current aspiration but DON'T go below BATNA.
- Opponent seemed generous → soft-type: claim slightly more.

## History Context
{history_context}

## Response Format
Return ONLY valid JSON:
{{"allocation_self": [items_you_keep], "allocation_other": [items_for_opponent], "reason": "brief"}}

Constraints:
- allocation_self[i] + allocation_other[i] = quantities[i] = {quantities}
- All integers >= 0
- sum(valuation_self[i] * allocation_self[i]) >= {batna_self}"""


ACCEPT_SYSTEM = """You are an expert negotiator deciding whether to accept or reject an offer.

## Game Rules
- 3 item types with quantities {quantities}.
- Your valuations private. Opponent's unknown.
- If reject: you counter-offer next round, all future payoffs discounted by gamma={discount}.
- If no deal by final round: both get BATNA.

## Your Situation
- Your valuations: {valuations_self}
- Your BATNA: {batna_self}
- Discount: {discount}
- Round: {round_index} of {max_rounds}
- Rounds remaining after this: {rounds_left}

## The Offer
- Items you receive: {offer_allocation}
- Value to YOU: {offer_value}
- Items opponent keeps: {counter_allocation}

## CRITICAL: Err on the Side of ACCEPTING
The benchmark scores on combined welfare (UW, NW, NWA). A rejected-then-no-deal outcome is TERRIBLE for your score — both players drop to BATNA, killing UW%. Accepting a mediocre-but-positive offer almost always beats walking away.

## Decision Rule (follow in order)
1. **Final round** (rounds_left == 0): ACCEPT if offer_value >= BATNA. No downside.
2. **offer_value >= 1.10 * BATNA**: ACCEPT. You're strictly better than your fallback, and counters risk rejection cascades.
3. **offer_value < BATNA and offer_value < 0.85 * BATNA**: REJECT. Clearly exploitative.
4. **Borderline case** (offer between 0.85*BATNA and 1.10*BATNA):
   - If rounds_left <= 1: ACCEPT (limited time to recover via counter).
   - Otherwise: reject only if you could realistically counter-propose something yielding offer_value * {acceptance_boost:.2f} or more (i.e., recovering the discount loss).
5. **Never reject just because you think you "deserve more"** unless you have concrete reason (e.g., opponent's prior offer was meaningfully better).

## Expected Value of Rejection (EV calculation)
- If you reject, next round you counter-propose (or opponent walks).
- Best plausible counter value ≈ {expected_counter}, discounted to {discounted_expected_counter}.
- Risk of opponent rejecting your counter: moderate.
- So EV(reject) ≈ max(BATNA, 0.7 * {discounted_expected_counter}).
- ACCEPT if offer_value >= EV(reject).

## History Context
{history_context}

## Response Format
Return ONLY valid JSON:
{{"accept": true or false, "reason": "brief"}}"""


def _format_history(history: list[dict]) -> str:
    """Format history into a readable context for the LLM."""
    if not history:
        return "No prior rounds yet — no opponent signal."

    lines = ["Prior turns in THIS game:"]
    for h in history:
        turn = h.get("turn", "?")
        action = h.get("action", "?")
        if action == "propose":
            alloc = h.get("allocation_self") or []
            lines.append(f"  Round {turn}: I proposed to keep {alloc}")
        elif action == "accept_or_reject":
            decided = "ACCEPTED" if h.get("accept") else "REJECTED"
            offer = h.get("offer_to_me") or []
            lines.append(f"  Round {turn}: opponent offered me {offer}, I {decided}")
        elif action == "opponent_proposed":
            offer = h.get("offer_to_me") or []
            lines.append(f"  Round {turn}: opponent proposed giving me {offer}")

    # Opponent type inference
    hints = []
    opponent_offers = [h.get("offer_to_me") for h in history if h.get("offer_to_me")]
    if opponent_offers:
        total_given = sum(sum(o) for o in opponent_offers) / len(opponent_offers)
        if total_given <= 3:
            hints.append("Opponent appears TOUGH (minimal offers). Concede less; push just above their likely BATNA.")
        elif total_given >= 6:
            hints.append("Opponent appears SOFT (generous offers). Claim more aggressively.")
        else:
            hints.append("Opponent appears MODERATE (aspiration-like). Standard gradual concession.")
    if hints:
        lines.append("")
        lines.append("INFERRED OPPONENT TYPE:")
        lines.extend("  " + h for h in hints)

    return "\n".join(lines)


def build_propose_prompt(obs, history: list[dict] | None = None) -> str:
    # Opponent BATNA is uniform[1, opponent_total_value]. Since opponent values are unknown
    # but drawn from the same distribution, expected opponent total ≈ 50 * sum(quantities).
    max_opponent_total_est = 50 * sum(obs.quantities) * 2  # conservative upper bound
    expected_opp_batna = max_opponent_total_est // 2
    # Target: give opponent expected value ~1.3x their expected BATNA for safety.
    safe_opp_target = int(expected_opp_batna * 1.3)
    self_mean_val = sum(obs.valuations_self) // max(1, len(obs.valuations_self))

    return PROPOSE_SYSTEM.format(
        quantities=obs.quantities,
        discount=obs.discount,
        valuations_self=obs.valuations_self,
        total_value=obs.total_value,
        batna_self=obs.batna_self,
        round_index=obs.round_index,
        max_rounds=obs.max_rounds,
        max_opponent_total_est=max_opponent_total_est,
        expected_opp_batna=expected_opp_batna,
        expected_opp_value_per_unit=50,
        safe_opp_target=safe_opp_target,
        self_mean_val=self_mean_val,
        history_context=_format_history(history or []),
    )


def build_accept_prompt(obs, history: list[dict] | None = None) -> str:
    offer_alloc = obs.pending_offer_allocation or []
    counter_alloc = (obs.pending_offer or {}).get("offer_allocation_self", []) if obs.pending_offer else []

    if offer_alloc:
        offer_val = sum(v * a for v, a in zip(obs.valuations_self, offer_alloc))
    elif obs.offer_value is not None:
        offer_val = obs.offer_value
    else:
        offer_val = 0

    expected_counter = int(obs.total_value * 0.6)
    discounted_expected_counter = int(expected_counter * obs.discount)
    # Boost factor: how much MORE a counter must yield to justify rejection
    acceptance_boost = 1.0 / max(obs.discount, 0.5)

    return ACCEPT_SYSTEM.format(
        quantities=obs.quantities,
        discount=obs.discount,
        valuations_self=obs.valuations_self,
        batna_self=obs.batna_self,
        round_index=obs.round_index,
        max_rounds=obs.max_rounds,
        rounds_left=obs.max_rounds - obs.round_index,
        offer_allocation=offer_alloc,
        offer_value=offer_val,
        counter_allocation=counter_alloc,
        expected_counter=expected_counter,
        discounted_expected_counter=discounted_expected_counter,
        acceptance_boost=acceptance_boost,
        history_context=_format_history(history or []),
    )
