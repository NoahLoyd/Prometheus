# tools/tool_manager.py

from typing import Dict, Type
from tools.base_tool import BaseTool

class ToolManager:
    """
    Manages registration and invocation of modular tools for the AGI system.
    """

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool_instance: BaseTool):
        """
        Registers a tool by its .name attribute (lowercased).
        """
        if not hasattr(tool_instance, 'name') or not isinstance(tool_instance.name, str):
            raise ValueError(f"Tool {tool_instance} must have a string 'name' attribute.")
        name = tool_instance.name.strip().lower()
        if not name:
            raise ValueError("Tool name must be a non-empty string.")
        self.tools[name] = tool_instance

    def call_tool(self, name: str, query: str) -> str:
        """
        Calls a registered tool by name with the provided query.
        """
        name = name.strip().lower()
        tool = self.tools.get(name)
        if not tool:
            # Optionally, add aliases here if necessary
            available = ', '.join(self.tools.keys())
            return f"Tool '{name}' not found. Available tools: {available if available else 'none'}"
        try:
            return tool.run(query)
        except Exception as e:
            return f"Error running tool '{name}': {e}"
