# tools/tool_manager.py

class ToolManager:
    """
    A class to manage and run tools using the 'tool_name: input' format.
    """

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool_name, tool_instance):
        self.tools[tool_name.lower()] = tool_instance

    def call_tool(self, tool_name, query):
        tool = self.tools.get(tool_name.lower())
        if not tool:
            return f"Error: Tool '{tool_name}' not found."
        if not hasattr(tool, "run"):
            return f"Error: Tool '{tool_name}' does not have a 'run' method."
        return tool.run(query)

    def run_tool(self, command):
        if ":" not in command:
            return "Invalid command format. Use 'tool_name: input'."

        parts = command.split(":", 1)
        if len(parts) != 2:
            return "Command must follow the format 'tool_name: input'."

        tool_name, query = parts
        return self.call_tool(tool_name.strip(), query.strip())
