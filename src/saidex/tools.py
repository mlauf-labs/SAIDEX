"""Caller-supplied tools for the agent loop (extract_with_tools / run_agent_loop).

A :class:`Tool` wraps an async handler function with the Pydantic parameter schema
that describes the tool's arguments.  The lib converts each ``Tool`` into an
OpenAI-style function definition that is bound to the LLM via ``bind_tools``.

Example::

    from pydantic import BaseModel, Field
    from saidex.tools import Tool

    class CreateFolderArgs(BaseModel):
        name: str = Field(description="Folder name")
        parent_id: str | None = Field(default=None, description="Parent folder id")

    async def _create_folder_handler(name: str, parent_id: str | None = None) -> dict:
        # ... real implementation ...
        return {"folder_id": "abc123", "path": f"Root/{name}"}

    tool = Tool(
        name="create_folder",
        description="Create a new folder and return its id and full path.",
        parameters=CreateFolderArgs,
        handler=_create_folder_handler,
    )
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class Tool:
    """A caller-supplied tool the LLM may invoke during the agent loop.

    Attributes:
        name: The function name exposed to the LLM (must be a valid identifier).
        description: A clear description of what this tool does; shown to the LLM.
        parameters: A Pydantic ``BaseModel`` subclass whose fields define the
            tool's arguments.  The JSON Schema is generated automatically.
        handler: An *async* callable that receives the validated argument values
            as keyword arguments (matching the ``parameters`` field names) and
            returns a result that will be serialised to a string and appended as
            a :class:`~langchain_core.messages.ToolMessage` in the conversation.
    """

    name: str
    description: str
    parameters: type[BaseModel]
    handler: Callable[..., Awaitable[Any]]

    def to_openai_tool(self) -> dict[str, Any]:
        """Return an OpenAI-style function definition for ``bind_tools``.

        Uses the ``parameters`` Pydantic schema to build the ``parameters``
        section; overrides the top-level name / description with the
        ``Tool``-level values so callers have full control over naming.
        """
        schema = dict(self.parameters.model_json_schema())
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    async def execute(self, raw_args: dict[str, Any]) -> str:
        """Validate *raw_args* against ``parameters``, run the handler, return result.

        Args:
            raw_args: Raw argument dict from the LLM tool call.

        Returns:
            A string (JSON or plain text) to use as the ``ToolMessage`` content.
            Handler exceptions are caught and returned as error strings so the
            LLM can self-correct rather than crashing the whole pipeline.
        """
        from .utils import create_instance_safe  # local import to avoid circular deps

        args_model, error_text = create_instance_safe(self.parameters, **raw_args)
        if error_text or args_model is None:
            return f"Invalid arguments for tool '{self.name}': {error_text}"

        try:
            result = await self.handler(**args_model.model_dump())
        except Exception as exc:  # noqa: BLE001
            return f"Tool '{self.name}' raised an error: {exc}"

        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)
