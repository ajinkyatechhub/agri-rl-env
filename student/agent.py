# ------------------------------------------------------------
# THE STUDENT: an AI (Google Gemini, or any OpenAI-compatible
# service) that looks at the farm through its tools, decides what
# to do, and records every action it takes.
# ------------------------------------------------------------
import json
import os
from contextlib import AsyncExitStack

from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1, TRAJECTORY_V1, AgentConfig, AgentEnvAgent, AgentIdentity,
    TaskRequest, TaskResult, TextPart, Usage, a2a_agent,
)

# Each conversation's messages so far, so a later message continues it.
CONVERSATIONS: dict[str, list] = {}

# Some AI services (e.g. Gemini) reject a few JSON-schema keywords; remove them.
_DROP = {"$schema", "additionalProperties", "title"}


def _clean(schema):
    if isinstance(schema, dict):
        return {k: _clean(v) for k, v in schema.items() if k not in _DROP}
    if isinstance(schema, list):
        return [_clean(v) for v in schema]
    return schema


class StudentConfig(AgentConfig):
    model: str = "gemini-3.5-flash-lite"    # free-tier Gemini model
    system_prompt: str = "You work in the systems you are given. Use their tools, follow their rules, and say what you did."
    max_turns: int = 30


@a2a_agent(
    identity=AgentIdentity(name="student", description="An AI that works a system through its tools.", version="1.0.0"),
    config=StudentConfig,
    extensions=(MCP_CONFIG_V1, TRAJECTORY_V1),
)
class StudentAgent(AgentEnvAgent):
    async def run(self, request: TaskRequest[StudentConfig]) -> TaskResult:
        config = request.config
        client = AsyncOpenAI(base_url=os.environ["LITELLM_BASE_URL"], api_key=os.environ["LITELLM_API_KEY"], max_retries=8)
        message = "\n".join(p.text for p in request.parts if isinstance(p, TextPart))
        history = CONVERSATIONS.setdefault(request.context_id, [])
        history.append({"role": "user", "content": message})
        turn = [{"type": "message", "text": message}]   # the record the teacher will read

        async with AsyncExitStack() as stack:
            # 1. Connect to the farm and collect its tool "buttons".
            sessions, tools = {}, []
            for server in request.mcp_servers.values():
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(server["url"], headers=server.get("headers"))
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                for t in (await session.list_tools()).tools:
                    sessions[t.name] = session
                    tools.append({"type": "function", "function": {
                        "name": t.name, "description": t.description or "", "parameters": _clean(t.inputSchema),
                    }})

            # 2. Think -> press a button -> see the result -> repeat, until done.
            for _ in range(config.max_turns):
                response = await client.chat.completions.create(
                    model=config.model,
                    messages=[{"role": "system", "content": config.system_prompt}] + history,
                    tools=tools,
                )
                msg = response.choices[0].message

                if not msg.tool_calls:                     # no button pressed = finished
                    answer = msg.content or ""
                    history.append({"role": "assistant", "content": answer})
                    turn.append({"type": "answer", "text": answer})
                    calls = sum(s["type"] == "tool_call" for s in turn)
                    return (
                        TaskResult.builder().succeeded().add_text(answer)
                        .usage(Usage(tool_call_count=calls))
                        .native_trajectory(format="student/v1", payload=turn)
                        .build()
                    )

                # Send the AI's message back exactly as received (keeps hidden extras such as
                # Gemini "thought signatures", which Gemini requires for multi-step tool use).
                assistant = msg.model_dump(exclude_none=True)
                assistant["content"] = msg.content or ""
                history.append(assistant)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = None
                    if name not in sessions:
                        output = json.dumps({"error": f"No tool named {name}"})
                    elif not isinstance(args, dict):
                        output = json.dumps({"error": "Arguments were not valid JSON"})
                    else:
                        result = await sessions[name].call_tool(name, args)
                        output = "\n".join(c.text for c in result.content if c.type == "text")
                    turn.append({"type": "tool_call", "tool": name, "input": args, "output": output})
                    history.append({"role": "tool", "tool_call_id": tc.id, "content": output})

        return TaskResult.failure("max_turns", f"Still calling tools after {config.max_turns} model calls")


if __name__ == "__main__":
    StudentAgent().serve()