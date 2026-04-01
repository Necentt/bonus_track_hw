from __future__ import annotations

from typing import TypedDict


class NegotiationState(TypedDict, total=False):
    raw_message: str
    observation: dict | None
    action_type: str  # "propose" | "accept_or_reject"
    response_json: str
    error: str | None
