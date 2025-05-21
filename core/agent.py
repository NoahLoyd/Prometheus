# core/agent.py

from typing import Optional, Sequence
from tools.tool_manager import ToolManager
from tools.base_tool import BaseTool

class PrometheusAgent:
    def __init__(
        self,
        tools: Optional[Sequence[BaseTool]] = None,
        tool_manager: Optional[ToolManager] = None
    ):
        self.tool_manager: ToolManager = tool_manager if tool_manager is not None else ToolManager()
        if tools:
            for tool in tools:
                self.register_named_tool(tool.name, tool)

    def register_named_tool(self, name: str, tool_instance: BaseTool):
        """
        Registers a tool instance with the ToolManager.
        """
        if hasattr(tool_instance, "run"):
            self.tool_manager.register_tool(tool_instance)
        else:
            raise ValueError(f"Tool '{name}' must implement a 'run' method.")

    def run(self, command: str) -> str:
        """
        Handle commands like 'tool_name: argument string'.
        """
        try:
            parts = command.split(":", 1)  # Only split at the first colon
            if len(parts) != 2 or not parts[0].strip():
                return "Invalid command format. Tool name cannot be empty or missing a ':' separator."
            tool_name, query = parts
            tool_name = tool_name.strip().lower()
            query = query.strip()
            try:
                return self.tool_manager.call_tool(tool_name, query)
            except Exception as tool_exc:
                return f"Tool '{tool_name}' failed to execute: {tool_exc}"
        except Exception as exc:
            return f"Agent error: {exc}"

    def handle_command(self, command: str) -> str:
        """
        Alias for run() to support legacy/test code.
        """
        return self.run(command)
