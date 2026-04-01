from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, model_validator


QUANTITIES = [7, 4, 1]


class Observation(BaseModel):
    """Parsed observation from the green agent's message."""
    pair: str = ""
    game_index: int = 0
    role: str = ""  # "row" or "col"
    valuations_self: list[int] = []
    batna_self: int = 0
    discount: float = 0.98
    max_rounds: int = 5
    quantities: list[int] = QUANTITIES
    round_index: int = 1
    player_index: int = 0
    action: str = ""  # "propose" or "ACCEPT_OR_REJECT"
    pending_offer: dict[str, Any] | None = None
    offer_value: int | None = None
    batna_value: int | None = None
    value_cap: int = 100

    @property
    def total_value(self) -> int:
        return sum(v * q for v, q in zip(self.valuations_self, self.quantities))

    @property
    def is_propose(self) -> bool:
        return self.action.lower() == "propose"

    @property
    def is_accept_or_reject(self) -> bool:
        return self.action.lower() in ("accept_or_reject", "accept or reject")

    @property
    def pending_offer_allocation(self) -> list[int] | None:
        """What the opponent is offering us (allocation_other from their perspective)."""
        if not self.pending_offer:
            return None
        return self.pending_offer.get("offer_allocation_other")


class ProposalResponse(BaseModel):
    """Response for a PROPOSE action."""
    allocation_self: list[int]
    allocation_other: list[int] | None = None
    reason: str = ""

    @model_validator(mode="after")
    def validate_allocations(self) -> ProposalResponse:
        quantities = QUANTITIES
        if self.allocation_other is None:
            self.allocation_other = [
                q - s for q, s in zip(quantities, self.allocation_self)
            ]
        for i, (s, o, q) in enumerate(zip(self.allocation_self, self.allocation_other, quantities)):
            if s < 0 or o < 0:
                raise ValueError(f"Allocation for item {i} is negative")
            if s + o != q:
                raise ValueError(f"Item {i}: {s} + {o} != {q}")
        return self


class AcceptResponse(BaseModel):
    """Response for an ACCEPT_OR_REJECT action."""
    accept: bool
    reason: str = ""


def parse_observation_from_text(text: str) -> Observation:
    """Extract the observation JSON from the green agent's message text."""
    # Try to find JSON block after "Observation:" marker
    patterns = [
        r'Observation:\s*```json\s*(\{.*?\})\s*```',
        r'Observation:\s*(\{.*?\})',
        r'```json\s*(\{[^`]*?\})\s*```',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return Observation(**data)
            except (json.JSONDecodeError, Exception):
                continue

    # Fallback: find the largest JSON object in the text
    json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    for obj_str in sorted(json_objects, key=len, reverse=True):
        try:
            data = json.loads(obj_str)
            if "action" in data or "valuations_self" in data:
                return Observation(**data)
        except (json.JSONDecodeError, Exception):
            continue

    raise ValueError(f"Could not parse observation from message: {text[:200]}...")


def determine_action_type(text: str) -> str:
    """Determine whether the message is asking for PROPOSE or ACCEPT_OR_REJECT."""
    text_lower = text.lower()
    if "accept_or_reject" in text_lower or "action: accept_or_reject" in text_lower:
        return "accept_or_reject"
    if "action: propose" in text_lower or '"action": "propose"' in text_lower:
        return "propose"
    if "accept" in text_lower and "reject" in text_lower:
        return "accept_or_reject"
    return "propose"
