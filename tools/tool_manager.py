# tools/tool_manager.py

class ToolManager:
    """
    A class to manage and call tools using the 'tool_name: input' format.
    """

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool_name: str, tool_instance):
        """
        Registers a tool instance under a specific name.
        """
        if not hasattr(tool_instance, "run"):
            raise ValueError(f"Tool '{tool_name}' must implement a 'run' method.")
        self.tools[tool_name.lower()] = tool_instance

    def call_tool(self, tool_name: str, query: str) -> str:
        """
        Calls a registered tool with the given query.
        """
        tool = self.tools.get(tool_name.lower())
        if not tool:
            return f"Error: Tool '{tool_name}' not found."

        try:
            return tool.run(query)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"
            
