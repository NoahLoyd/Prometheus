# tool_manager.py

import os
import importlib.util

class ToolManager:
    def __init__(self, tools_folder="tools"):
        self.tools_folder = tools_folder
        self.tools = {}
        self.load_tools()

    def load_tools(self):
        self.tools.clear()
        for filename in os.listdir(self.tools_folder):
            if filename.endswith(".py") and not filename.startswith("__"):
                tool_name = filename[:-3]
                path = os.path.join(self.tools_folder, filename)
                spec = importlib.util.spec_from_file_location(tool_name, path)
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    if hasattr(module, "tool"):
                        self.tools[tool_name] = module.tool
                except Exception as e:
                    print(f"Error loading tool {tool_name}: {e}")

    def get_tool(self, name):
        return self.tools.get(name)

    def list_tools(self):
        return list(self.tools.keys())

    def call_tool(self, name, input_text):
        tool = self.get_tool(name)
        if not tool:
            return f"Tool '{name}' not found."
        try:
            return tool.run(input_text)
        except Exception as e:
            return f"Error running tool '{name}': {e}"
