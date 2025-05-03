class ToolManager:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, func):
        """Register a new tool with a unique name."""
        self.tools[name] = func

    def list_tools(self):
        """List all registered tool names."""
        return list(self.tools.keys())

    def call_tool(self, name, *args, **kwargs):
        """Call a registered tool by name with given arguments."""
        if name in self.tools:
            return self.tools[name](*args, **kwargs)
        else:
            return f"Tool '{name}' not found."
