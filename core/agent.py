# core/agent.py

from typing import Optional, Sequence
from tools.tool_manager import ToolManager

class PrometheusAgent:
    def __init__(
        self,
        tools: Optional[Sequence] = None,
        tool_manager: Optional[ToolManager] = None
    ):
        self.tool_manager = tool_manager if tool_manager is not None else ToolManager()

        # Register core tools with explicit names
        if tools:
            for tool in tools:
                self.register_named_tool(tool.name.lower(), tool)

    def register_named_tool(self, name, tool_instance):
        """
        Registers a tool with a specific name and its instance.
        """
        if hasattr(tool_instance, "run"):
            self.tool_manager.register_tool(name, tool_instance)
        else:
            raise ValueError(f"Tool '{name}' must implement a 'run' method.")

    def run(self, command: str) -> str:
        """
        Handle commands like 'tool_name: save: message'.
        """
        parts = command.split(":", 1)  # Only split at the first colon
        if len(parts) != 2 or not parts[0].strip():
            return "Invalid command format. Tool name cannot be empty."

        tool_name, query = parts
        return self.tool_manager.call_tool(tool_name.strip().lower(), query.strip())
