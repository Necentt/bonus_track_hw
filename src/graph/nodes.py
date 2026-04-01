"""LangGraph nodes for the negotiation pipeline."""

from __future__ import annotations

import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.state import NegotiationState
from strategy.models import (
    Observation,
    ProposalResponse,
    AcceptResponse,
    parse_observation_from_text,
    determine_action_type,
    QUANTITIES,
)
from strategy.prompts import build_propose_prompt, build_accept_prompt
from strategy.heuristics import aspiration_propose, aspiration_accept_or_reject
from llm.client import get_llm


async def parse_observation(state: NegotiationState) -> NegotiationState:
    """Parse the green agent's message into a structured observation."""
    raw = state["raw_message"]
    try:
        obs = parse_observation_from_text(raw)
        action_type = "propose" if obs.is_propose else "accept_or_reject"
        if not action_type or action_type not in ("propose", "accept_or_reject"):
            action_type = determine_action_type(raw)
        return {
            "observation": obs.model_dump(),
            "action_type": action_type,
            "error": None,
        }
    except Exception as e:
        action_type = determine_action_type(raw)
        return {
            "observation": None,
            "action_type": action_type,
            "error": f"Parse error: {e}",
        }


async def strategize(state: NegotiationState) -> NegotiationState:
    """Use LLM to decide on a negotiation action."""
    obs_data = state.get("observation")
    if not obs_data:
        return {"error": "No observation to strategize on"}

    obs = Observation(**obs_data)
    action_type = state["action_type"]

    try:
        llm = get_llm()

        if action_type == "propose":
            system_prompt = build_propose_prompt(obs)
            user_msg = f"Propose an allocation for round {obs.round_index}. Your valuations: {obs.valuations_self}, BATNA: {obs.batna_self}"

            response = await llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ])

            # Parse the LLM response
            resp_text = response.content.strip()
            resp_data = _extract_json(resp_text)
            proposal = ProposalResponse(**resp_data)

            # Validate: self value >= BATNA
            self_value = sum(v * a for v, a in zip(obs.valuations_self, proposal.allocation_self))
            if self_value < obs.batna_self and obs.round_index < obs.max_rounds:
                # LLM proposed a bad deal, use heuristic
                return {"error": f"LLM proposal value {self_value} < BATNA {obs.batna_self}"}

            return {
                "response_json": json.dumps({
                    "allocation_self": proposal.allocation_self,
                    "allocation_other": proposal.allocation_other,
                    "reason": proposal.reason,
                }),
                "error": None,
            }
        else:
            system_prompt = build_accept_prompt(obs)
            offer_alloc = obs.pending_offer_allocation or []
            user_msg = f"Accept or reject? Offered items: {offer_alloc}, your BATNA: {obs.batna_self}"

            response = await llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ])

            resp_text = response.content.strip()
            resp_data = _extract_json(resp_text)
            accept_resp = AcceptResponse(**resp_data)

            return {
                "response_json": json.dumps({
                    "accept": accept_resp.accept,
                    "reason": accept_resp.reason,
                }),
                "error": None,
            }

    except Exception as e:
        return {"error": f"LLM error: {e}"}


async def format_response(state: NegotiationState) -> NegotiationState:
    """Validate and finalize the response JSON."""
    resp_json = state.get("response_json", "")
    if not resp_json:
        return {"error": "No response to format"}

    try:
        data = json.loads(resp_json)

        # Validate proposal allocations if present
        if "allocation_self" in data:
            alloc_self = data["allocation_self"]
            alloc_other = data.get("allocation_other")
            if alloc_other is None:
                alloc_other = [q - s for q, s in zip(QUANTITIES, alloc_self)]
                data["allocation_other"] = alloc_other

            for i, (s, o, q) in enumerate(zip(alloc_self, alloc_other, QUANTITIES)):
                if s < 0 or o < 0 or s + o != q:
                    return {"error": f"Invalid allocation at item {i}: {s}+{o}!={q}"}

        return {"response_json": json.dumps(data), "error": None}
    except (json.JSONDecodeError, Exception) as e:
        return {"error": f"Format error: {e}"}


async def fallback(state: NegotiationState) -> NegotiationState:
    """Use heuristic fallback when LLM fails."""
    obs_data = state.get("observation")
    action_type = state.get("action_type", "propose")

    if obs_data:
        obs = Observation(**obs_data)
    else:
        # Last resort: return a safe default
        if action_type == "accept_or_reject":
            return {"response_json": json.dumps({"accept": False, "reason": "Fallback: no observation"})}
        return {"response_json": json.dumps({
            "allocation_self": [4, 2, 1],
            "allocation_other": [3, 2, 0],
            "reason": "Fallback: no observation",
        })}

    if action_type == "propose":
        proposal = aspiration_propose(obs)
        return {"response_json": json.dumps({
            "allocation_self": proposal.allocation_self,
            "allocation_other": proposal.allocation_other,
            "reason": proposal.reason,
        }), "error": None}
    else:
        accept_resp = aspiration_accept_or_reject(obs)
        return {"response_json": json.dumps({
            "accept": accept_resp.accept,
            "reason": accept_resp.reason,
        }), "error": None}


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response text."""
    import re

    # Try direct parse
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object in text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        return json.loads(match.group())

    raise ValueError(f"Could not extract JSON from: {text[:200]}")
