# tools/tool_manager.py

from typing import Dict, Any
from tools.base_tool import BaseTool

class ToolManager:
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool_instance: BaseTool):
        """
        Registers a tool by its .name attribute (lowercased).
        """
        if not hasattr(tool_instance, 'name'):
            raise ValueError("Tool instance must have a 'name' attribute.")
        name = tool_instance.name.strip().lower()
        self.tools[name] = tool_instance

    def call_tool(self, name: str, query: str) -> str:
        """
        Calls the tool with the provided name and passes the query string.
        """
        name = name.strip().lower()
        tool = self.tools.get(name)
        if not tool:
            # Optionally, support aliasing: e.g. map "calculate" to "calculator"
            aliases = {"calculate": "calculator"}
            actual_name = aliases.get(name)
            if actual_name:
                tool = self.tools.get(actual_name)
        if not tool:
            available = ', '.join(self.tools.keys())
            return f"Tool '{name}' not found. Available tools: {available if available else 'none'}"
        try:
            return tool.run(query)
        except Exception as e:
            return f"Error running tool '{name}': {e}"
