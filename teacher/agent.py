# ------------------------------------------------------------
# THE TEACHER: an AI (Google Gemini, or any OpenAI-compatible
# service) that reads the student's recorded actions and marks
# each line of the answer key true or false.
# ------------------------------------------------------------
import json
import os
from pathlib import Path

from openai import AsyncOpenAI

from agentenv_protocol.a2a_agent import (
    TRAJECTORY_V1, AgentConfig, AgentEnvAgent, AgentIdentity,
    TaskRequest, TaskResult, TextPart, a2a_agent,
)

READ_FILE = {"type": "function", "function": {
    "name": "read_file",
    "description": "Read a file the grading prompt points you at, such as the agent's trajectory, or list a directory.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Absolute path of the file or directory."}},
        "required": ["path"],
    },
}}


class TeacherConfig(AgentConfig):
    model: str = "gemini-3.5-flash-lite"    # free-tier Gemini model
    system_prompt: str = (
        "You grade another agent's work against a rubric. Read the files the prompt points you at "
        "before you decide, and answer with only the JSON the prompt asks for: raw JSON, "
        "with no markdown code fences and no extra text."
    )
    max_turns: int = 10


def _strip_fences(text: str) -> str:
    """Remove markdown code fences like ```json ... ``` that some AIs wrap around JSON."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rstrip()
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _read(path_text: str) -> str:
    path = Path(path_text)
    if path.is_dir():
        return "\n".join(sorted(str(p) for p in path.iterdir()))
    if path.is_file():
        return path.read_text()
    return f"Nothing at {path}"


@a2a_agent(
    identity=AgentIdentity(name="teacher", description="An AI that grades an agent's work against a rubric.", version="1.0.0"),
    config=TeacherConfig,
    extensions=(TRAJECTORY_V1,),
)
class TeacherAgent(AgentEnvAgent):
    async def run(self, request: TaskRequest[TeacherConfig]) -> TaskResult:
        config = request.config
        client = AsyncOpenAI(base_url=os.environ["LITELLM_BASE_URL"], api_key=os.environ["LITELLM_API_KEY"], max_retries=8)
        prompt = "\n".join(p.text for p in request.parts if isinstance(p, TextPart))
        messages = [{"role": "system", "content": config.system_prompt}, {"role": "user", "content": prompt}]
        reads = []
        for _ in range(config.max_turns):
            response = await client.chat.completions.create(
                model=config.model, messages=messages, tools=[READ_FILE],
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                verdict = _strip_fences(msg.content or "")
                return (
                    TaskResult.builder().succeeded().add_text(verdict)
                    .native_trajectory(format="teacher/v1", payload=reads).build()
                )
            # Send the AI's message back exactly as received (keeps hidden extras such as
            # Gemini "thought signatures", which Gemini requires for multi-step tool use).
            assistant = msg.model_dump(exclude_none=True)
            assistant["content"] = msg.content or ""
            messages.append(assistant)
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    content = _read(args["path"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    content = "Error: call read_file with a JSON object like {\"path\": \"/tmp/file.json\"}"
                reads.append({"tool": tc.function.name, "chars": len(content)})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        return TaskResult.failure("max_turns", f"Still reading files after {config.max_turns} turns")


if __name__ == "__main__":
    TeacherAgent().serve()