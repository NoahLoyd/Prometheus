# tools/tool_manager.py

class ToolManager:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, tool_func):
        self.tools[name] = tool_func

    def use_tool(self, name, *args, **kwargs):
        if name not in self.tools:
            return f"Tool '{name}' not found."
        return self.tools[name](*args, **kwargs)

    def list_tools(self):
        return list(self.tools.keys())
