import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from graph.graph import build_negotiation_graph


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.graph = build_negotiation_graph()
        # History of turns in the CURRENT game (Agent is per-context_id, so per-game).
        # Each entry: {"turn", "action", "allocation_self"?, "offer_to_me"?, "accept"?}
        self.history: list[dict] = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Analyzing negotiation state...")
        )

        result = await self.graph.ainvoke({
            "raw_message": input_text,
            "history": list(self.history),
        })

        response_json = result.get("response_json", '{"accept": false, "reason": "Agent error"}')

        # Record this turn into history
        self._record_turn(result, response_json)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_json))],
            name="negotiation_response",
        )

    def _record_turn(self, result: dict, response_json: str) -> None:
        obs = result.get("observation") or {}
        action_type = result.get("action_type", "")
        turn = obs.get("round_index", len(self.history) + 1)

        # If opponent made a pending offer, record that first (before our response)
        pending = obs.get("pending_offer") or {}
        if pending and pending.get("offer_allocation_other"):
            self.history.append({
                "turn": turn,
                "action": "opponent_proposed",
                "offer_to_me": pending.get("offer_allocation_other"),
                "opponent_kept": pending.get("offer_allocation_self"),
            })

        try:
            resp = json.loads(response_json)
        except Exception:
            return

        if action_type == "propose" and "allocation_self" in resp:
            self.history.append({
                "turn": turn,
                "action": "propose",
                "allocation_self": resp.get("allocation_self"),
                "allocation_other": resp.get("allocation_other"),
            })
        elif action_type == "accept_or_reject" and "accept" in resp:
            self.history.append({
                "turn": turn,
                "action": "accept_or_reject",
                "accept": resp.get("accept"),
                "offer_to_me": pending.get("offer_allocation_other"),
            })
