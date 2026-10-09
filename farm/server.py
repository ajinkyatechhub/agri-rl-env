# ------------------------------------------------------------
# THE WORLD: a tiny farm with fields and a rule book.
# The AI agent can look at fields and decide: spray or healthy.
# ------------------------------------------------------------
import json
from pathlib import Path

from agentenv_protocol import (
    AgentEnvEnvironment, DataPart, FilePart,
    add_data, environment_card, get_data, reset_data, tool,
)


@environment_card(name="farm")          # this world's name is "farm"
class FarmEnv(AgentEnvEnvironment):
    """A tiny farm: a few fields, a rule book, and two actions."""

    def __init__(self) -> None:
        self.rules = []                  # the rule book
        self.fields = {}                 # the fields, by id like "F-1"

    # ===== PART A: data control (used by AgentEnv, hidden from the AI) =====

    @reset_data                          # "make the world empty"
    async def _reset(self) -> None:
        self.rules = []
        self.fields = {}

    @add_data                            # "fill the world from farm.json"
    async def _add(self, parts: list) -> None:
        for part in parts:
            if isinstance(part, DataPart):
                data = part.data
            elif isinstance(part, FilePart):
                data = json.loads(Path(part.file.uri.removeprefix("file://")).read_text())
            else:
                continue
            self.rules.extend(data.get("rules", []))
            for f in data.get("fields", []):
                self.fields[f["field_id"]] = {"status": "pending", **f}

    @get_data                            # "show me the world right now"
    async def _state(self) -> list:
        return [DataPart(data={"fields": self.fields})]

    # ===== PART B: tools (the buttons the AI can press) =====

    @tool(name="{environment_name}_read_rules")
    async def read_rules(self) -> dict:
        """Read the farm's rule book. Do this first."""
        return {"rules": self.rules}

    @tool(name="{environment_name}_list_fields")
    async def list_fields(self) -> dict:
        """List every field with its crop and current status."""
        return {"fields": [
            {"field_id": k, "crop": v["crop"], "status": v["status"]}
            for k, v in self.fields.items()
        ]}

    @tool(name="{environment_name}_check_field")
    async def check_field(self, field_id: str) -> dict:
        """Look at one field closely: its disease and how bad it is."""
        return self.fields.get(field_id) or {"error": f"No field {field_id}"}

    @tool(name="{environment_name}_spray")
    async def spray(self, field_id: str, note: str) -> dict:
        """Spray medicine on a field. Write why in the note."""
        return self._decide(field_id, "sprayed", note)

    @tool(name="{environment_name}_mark_healthy")
    async def mark_healthy(self, field_id: str, note: str) -> dict:
        """Mark a field as fine, no spray needed. Write why in the note."""
        return self._decide(field_id, "healthy", note)

    # ===== PART C: the world's own safety rules =====

    def _decide(self, field_id: str, status: str, note: str) -> dict:
        f = self.fields.get(field_id)
        if f is None:
            return {"error": f"No field {field_id}"}
        if f["status"] != "pending":
            return {"error": f"{field_id} is already {f['status']}"}
        if not note.strip():
            return {"error": "Please write a note"}
        f["status"] = status
        f["note"] = note
        return {"field_id": field_id, "status": status}


if __name__ == "__main__":
    FarmEnv().serve()                    # start the world