"""Negotiation system prompts for the LLM."""


PROPOSE_SYSTEM = """You are an expert negotiator in a multi-round bargaining game. You must propose how to split items between yourself and your opponent.

## Game Rules
- 3 item types with quantities {quantities}.
- Both players have PRIVATE valuations. You know yours, not opponent's.
- Each round, one player proposes. The other accepts or rejects.
- If rejected, next round begins, all future values discounted by gamma={discount}.
- If no deal by final round, both get BATNA (outside offer).
- Your payoff = sum(valuation[i] * items_you_keep[i]) * gamma^(round-1).

## Your Situation
- Your valuations (per unit): {valuations_self}
- Your total possible value: {total_value}
- Your BATNA: {batna_self}
- Discount: {discount}
- Round: {round_index} of {max_rounds}

## CRITICAL: Pareto Efficiency (HIGH IMPACT ON SCORE)
The benchmark scores heavily on TOTAL welfare, not just your share. Key insight:
- Valuations are drawn UNIFORMLY at random from [1, 99]. So on average, opponent values each item SIMILARLY to you BUT WITH NOISE.
- However, the expected opponent valuation of any item is ~50 per unit. Quantities are [7, 4, 1], so expected opponent total ~ 7*50 + 4*50 + 1*50 = 600.
- Items where YOUR valuation is HIGH (>50) are probably relatively more valuable to you than opponent.
- Items where YOUR valuation is LOW (<50) are probably relatively more valuable to opponent.
- TO MAXIMIZE TOTAL WELFARE: keep items you value HIGH, give away items you value LOW. This is win-win.

## Strategy Rules
1. NEVER propose a deal below your BATNA for yourself.
2. Rank your items by per-unit value. Keep all units of items you value highest. Give ALL units of items you value lowest.
3. For middle-value items, split based on aspiration level:
   - Round 1: aim for ~75-80% of total value for yourself
   - Round 2: ~65-70%
   - Round 3+: ~55-60%
   - Final round: just above your BATNA (threat of walk-away is gone)
4. Opponent needs to get enough VALUE that they'd accept. Their BATNA is unknown but on average around half of total. Give them at least ~35-45% in earlier rounds, ~45-55% in late rounds.
5. If opponent has been HARSH in previous rounds (see history), they are "tough" type — concede less. If they've accepted freely or offered generously, they are "soft" — take more.

## History-Based Opponent Detection
{history_context}

## Response Format
Return ONLY valid JSON:
{{"allocation_self": [items_you_keep], "allocation_other": [items_for_opponent], "reason": "brief"}}

Constraints:
- allocation_self[i] + allocation_other[i] = quantities[i] = {quantities}
- All integers >= 0
- sum(valuation_self[i] * allocation_self[i]) >= {batna_self} (your BATNA)"""


ACCEPT_SYSTEM = """You are an expert negotiator deciding whether to accept or reject an offer.

## Game Rules
- 3 item types with quantities {quantities}.
- Your valuations private. Opponent's unknown.
- If reject: you counter-offer next round, all future values discounted by gamma={discount}.
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

## Decision Logic (CRITICAL)
1. If offer_value >= BATNA and this is the FINAL round: ALWAYS ACCEPT. Rejecting means BATNA anyway, no upside.
2. If rounds_left == 0: ACCEPT if offer_value >= BATNA.
3. Expected value of rejecting:
   - Counter-offer next round worth ~{expected_counter} (your likely proposal * discount)
   - But opponent might reject your counter, cascading to BATNA at final round
   - Conservative EV of rejection ≈ max(BATNA, offer_value * {discount})
4. If offer_value >= EV of rejection: ACCEPT.
5. If offer_value is within 10% of your best feasible counter: ACCEPT (avoid needless escalation — hurts welfare metrics).
6. Only reject if you have STRONG reason to believe next round yields materially more.

## History-Based Opponent Detection
{history_context}

## Response Format
Return ONLY valid JSON:
{{"accept": true or false, "reason": "brief"}}"""


def _format_history(history: list[dict]) -> str:
    """Format history into a readable context for the LLM."""
    if not history:
        return "No prior rounds in this game yet — no opponent signal available."

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

    # Opponent type inference hints
    hints = []
    opponent_offers = [h.get("offer_to_me") for h in history if h.get("offer_to_me")]
    if opponent_offers:
        # Check if offers are stingy (total units given is low)
        total_given = sum(sum(o) for o in opponent_offers) / len(opponent_offers)
        if total_given <= 3:
            hints.append("Opponent appears TOUGH (minimal offers). Consider: concede less, but avoid walk-away — push just above their likely BATNA.")
        elif total_given >= 6:
            hints.append("Opponent appears SOFT (generous offers). Consider: claim more aggressively.")
        else:
            hints.append("Opponent appears MODERATE (aspiration-like). Consider: standard gradual concession.")
    if hints:
        lines.append("")
        lines.append("INFERRED OPPONENT TYPE:")
        lines.extend("  " + h for h in hints)

    return "\n".join(lines)


def build_propose_prompt(obs, history: list[dict] | None = None) -> str:
    return PROPOSE_SYSTEM.format(
        quantities=obs.quantities,
        discount=obs.discount,
        valuations_self=obs.valuations_self,
        total_value=obs.total_value,
        batna_self=obs.batna_self,
        round_index=obs.round_index,
        max_rounds=obs.max_rounds,
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

    # Estimate what a counter-offer next round could yield
    expected_counter = int(obs.total_value * 0.6 * obs.discount)

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
        history_context=_format_history(history or []),
    )
