from __future__ import annotations

from dataclasses import dataclass

from .bus import Event, EventBus
from .intent import Intent


@dataclass(frozen=True)
class ActionResult:
    success: bool
    message: str


class ActionExecutor:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus

    def run(self, intent: Intent) -> ActionResult:
        payload = {"intent": intent.name, "slots": intent.slots}
        self.bus.publish(Event("action.run", payload))
        return ActionResult(success=True, message=f"Executed {intent.name}")
