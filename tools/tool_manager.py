# tools/tool_manager.py

class ToolManager:
    def __init__(self):
        self.tools = {}

    def register_tool(self, tool, func):
        """Register a tool using its class name as the key."""
        name = tool.__class__.__name__.lower()
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

    def run_tool(self, command):
        """Try running the command through all tools."""
        for name, tool_func in self.tools.items():
            try:
                result = tool_func(command)
                if result:
                    return result
            except Exception:
                pass
        return "No tool was able to process the command."
