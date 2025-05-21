# tools/tool_manager.py

from typing import Dict, Optional, Any, Protocol


class BaseTool(Protocol):
    """
    Protocol for all tools.
    Each tool must have a `name` attribute (str) and a `run(query: str) -> Any` method.
    """
    name: str

    def run(self, query: str) -> Any:
        ...


class ToolManager:
    """
    Manages tool registration and retrieval for modular AGI systems.
    """

    def __init__(self) -> None:
        self.tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        """
        Registers a tool instance.
        The tool is keyed by its lowercase name.
        Raises TypeError if the tool does not implement BaseTool.
        """
        if not isinstance(tool, BaseTool):
            # Fallback for Protocol when used with isinstance. You may want to
            # use abc.ABC for strict enforcement in production.
            if not hasattr(tool, "name") or not callable(getattr(tool, "run", None)):
                raise TypeError("Tool must implement 'name' (str) and 'run(query: str) -> Any'.")
        if not isinstance(tool.name, str):
            raise TypeError("Tool must have a string 'name' attribute.")
        key = tool.name.lower()
        self.tools[key] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Retrieves a tool by name (case-insensitive).
        Returns None if tool is not found.
        """
        if not isinstance(name, str):
            return None
        return self.tools.get(name.lower(), None)

    def call_tool(self, tool_name: str, query: str) -> Any:
        """
        Calls a registered tool's 'run' method with the given query.
        Raises KeyError if the tool is not found.
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            raise KeyError(f"Tool '{tool_name}' not found.")
        try:
            return tool.run(query)
        except Exception as e:
            raise RuntimeError(f"Error executing tool '{tool_name}': {e}") from e
