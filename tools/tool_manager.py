# tools/tool_manager.py

class ToolManager:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, func):
        """Register a tool with its handler function."""
        self.tools[name.lower()] = func

    def list_tools(self):
        return list(self.tools.keys())

    def call_tool(self, name, *args, **kwargs):
        if name.lower() in self.tools:
            return self.tools[name.lower()](*args, **kwargs)
        return f"Tool '{name}' not found."

    def run_tool(self, command):
        """Parse and run tool using 'tool_name: input' format."""
        if ":" not in command:
            return "Invalid command format. Use 'tool_name: input'"

        tool_name, query = command.split(":", 1)
        tool_name = tool_name.strip().lower()
        query = query.strip()

        return self.call_tool(tool_name, query)
