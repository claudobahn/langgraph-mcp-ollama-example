import asyncio
import sys
from datetime import datetime
from typing import cast, Any, Dict, AsyncIterator

from langchain_core.messages import (
    SystemMessage,
    BaseMessage,
    AIMessage,
    HumanMessage,
    ToolMessage, BaseMessageChunk,
)
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_SERVER_URL = "http://mcp-server:13744/mcp"
OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_MODEL = "qwen3:30b"
OLLAMA_TEMPERATURE = 0.8
OLLAMA_NUM_PREDICT = 4096

_last_prefix: str | None = None


def _extract_reasoning(msg: BaseMessage) -> str:
    """Try to extract model reasoning, if provided.

    Returns a string or empty string if none found.
    """
    try:
        addl = getattr(msg, "additional_kwargs", {}) or {}
        if isinstance(addl, dict):
            r = addl.get("reasoning_content")
            if isinstance(r, str) and r:
                return r
    except Exception:
        pass
    return ""


def _format_content(content) -> str:
    """Best-effort pretty formatting for message content which may be str or structured.

    Handles common cases where content is a list of blocks (text/tool_use/tool_result),
    otherwise falls back to string conversion.
    """
    try:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # LangChain sometimes stores content as a list of blocks
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    # Skip dedicated reasoning/thinking blocks here; handled separately
                    if btype in {"reasoning", "thinking"}:
                        continue
                    if btype == "text" and isinstance(block.get("text"), str):
                        parts.append(block["text"])
                    elif btype == "tool_use":
                        name = block.get("name", "tool")
                        args = block.get("input") or block.get("args") or {}
                        parts.append(f"[calls {name} {args}]")
                    elif btype == "tool_result":
                        name = block.get("name", "tool")
                        out = block.get("output")
                        out_str = out if isinstance(out, str) else str(out)
                        parts.append(f"[result from {name}: {out_str}]")
                    else:
                        parts.append(str(block))
                else:
                    parts.append(str(block))
            return "\n".join(p for p in parts if p)
        # Dict or other structured types
        return str(content)
    except Exception:
        return str(content)


def format_message(msg: BaseMessage) -> str:
    """Return a user-friendly, single-string representation of a message.

    Includes role labels and attempts to surface tool calls/results clearly.
    """
    global _last_prefix

    role = getattr(msg, "type", "message")
    prefix = {
        "human": "You",
        "ai": "Assistant",
        "AIMessageChunk": "Assistant",
        "tool": "Tool",
        "system": "System",
    }.get(role, role.capitalize())

    def maybe_prefix_line(line: str, include_prefix: bool) -> str:
        return f"\n{prefix}: {line}" if include_prefix else line

    reasoning = _extract_reasoning(msg)
    if reasoning:
        prefix += " (reasoning)"

    # Determine if we should print the prefix based on whether it changed
    include_prefix = prefix != _last_prefix

    lines: list[str] = []

    # AIMessage: show content and any tool calls
    if isinstance(msg, AIMessage):
        if reasoning:
            lines.append(maybe_prefix_line(reasoning, include_prefix))
            # Subsequent lines in the same message shouldn't repeat the prefix
            include_prefix = False
        text = _format_content(msg.content)
        if text:
            lines.append(maybe_prefix_line(text, include_prefix))
            include_prefix = False
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)

                lines.append(maybe_prefix_line(f"Calling tool -> {name} with args: {args}", include_prefix))
                include_prefix = False
        if lines:
            _last_prefix = prefix
        return "\n".join(lines)

    # Human message
    if isinstance(msg, HumanMessage):
        text = _format_content(msg.content)
        out = maybe_prefix_line(text, include_prefix)
        _last_prefix = prefix
        return out

    # Tool message
    if isinstance(msg, ToolMessage):
        name = getattr(msg, "name", "tool")
        label = f"[Tool {name}]"
        text = _format_content(msg.content)
        if include_prefix:
            result = f"\n{label} -> {text}" if text else label
        else:
            result = text
        _last_prefix = prefix
        return result

    # System message
    if isinstance(msg, SystemMessage):
        text = _format_content(msg.content)
        if include_prefix:
            out = f"[System] {text}" if text else "[System]"
        else:
            out = text
        _last_prefix = prefix
        return out

    # Try to include reasoning if present
    if reasoning:
        lines.append(maybe_prefix_line(f"Reasoning: {reasoning}", include_prefix))
        include_prefix = False

    # Generic BaseMessage or chunk fallback
    text = _format_content(getattr(msg, "content", ""))
    if text:
        lines.append(maybe_prefix_line(text, include_prefix))
    elif include_prefix:
        lines.append(maybe_prefix_line("", include_prefix))
    _last_prefix = prefix
    return "\n".join(l for l in lines if l != "")


def build_system_prompt(now: datetime | None = None) -> str:
    """Create the system prompt text with the current date/time embedded."""
    if now is None:
        now = datetime.now().astimezone()
    now_str = now.strftime("%A, %B %d, %Y %I:%M:%S %p %Z (%z)")
    return (
        f"""
            You are an AI assistant operating in a LangGraph ReAct loop with MCP tools.
            Current date/time: {now_str}

            Objectives:
            - Be helpful, accurate, and concise. Prefer short, direct answers unless more detail is requested.
            - Use MCP tools when they can improve accuracy or are necessary to obtain information.
            - Do not reveal internal chain-of-thought or tool call internals; summarize reasoning briefly instead.

            Language:
            - Reply in the user's language if it is clear from the input; otherwise default to English.

            Tool use policy:
            - Prefer the provided MCP tools for retrieval, calculations, file or system access.
            - If you need data you don't have, first check available tools. If tools fail or are unavailable, explain limitations succinctly.
            - When using tools, briefly note why you are using the tool and then provide the final answer after results are obtained.

            Output formatting:
            - Provide clean text without unnecessary code fences unless the user asks for code.
            - If you cite information obtained via tools, mention the tool name and key parameters or source identifiers when helpful.

            Safety and quality:
            - Avoid fabricating facts. If uncertain, say so and propose next steps.
            - Keep private keys, secrets, and system internals confidential.
            """
    )


def create_llm() -> ChatOllama:
    """Factory for the configured LLM."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        reasoning=True,
        validate_model_on_init=True,
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_NUM_PREDICT,
    )


async def stream_agent_messages(agent, user_prompt: str) -> None:
    """Stream messages from the agent to stdout using format_message."""
    async for (chunk, _metadata) in cast(
            AsyncIterator[tuple[BaseMessageChunk, Dict[str, Any]]],
            agent.astream({"messages": user_prompt}, stream_mode="messages"),
    ):
        print(format_message(chunk), end="", flush=True)
    print()


async def interact_with_assistant(user_prompt: str):
    async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            llm = create_llm()

            prompt_text = build_system_prompt()

            agent = create_react_agent(
                llm,
                tools,
                prompt=SystemMessage(content=prompt_text)
            )

            await stream_agent_messages(agent, user_prompt)


def read_user_prompt() -> str:
    """Read user prompt from stdin or interactive input, with a default fallback."""
    if sys.stdin is not None and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        try:
            text = input("Enter your prompt: ").strip()
        except EOFError:
            text = ""
    return text or "Demonstrate your tool usage."


if __name__ == "__main__":
    asyncio.run(interact_with_assistant(read_user_prompt()))
