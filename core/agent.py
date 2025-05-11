# core/agent.py

from tools.tool_manager import ToolManager

class PrometheusAgent:
    def __init__(self, tools=None):
        self.tool_manager = ToolManager()

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
        Handle commands like 'tool_name: input'.
        """
        if ":" not in command:
            return "Invalid command format. Use 'tool_name: input'."

        parts = command.split(":", 1)
        if len(parts) != 2 or not parts[0].strip():
            return "Invalid command format. Tool name cannot be empty."

        tool_name, query = parts
        return self.tool_manager.call_tool(tool_name.strip().lower(), query.strip())