import sys
import os

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

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working, new_agent_text_message("Analyzing negotiation state...")
        )

        result = await self.graph.ainvoke({"raw_message": input_text})

        response_json = result.get("response_json", '{"accept": false, "reason": "Agent error"}')

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_json))],
            name="negotiation_response",
        )
