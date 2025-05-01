# core/agent.py

from tools.file_tool import FileTool
from tools.internet_tool import fetch_summary
from memory.short_term import ShortTermMemory
from tools.tool_manager import ToolManager
from tools.calculator import calculate

class PrometheusAgent:
    def __init__(self, name="Prometheus", memory_limit=10):
        self.name = name
        self.memory = ShortTermMemory(limit=memory_limit)
        self.tool_manager = ToolManager()
        self._register_default_tools()

    def _register_default_tools(self):
        self.tool_manager.register_tool("calculator", calculate)
        self.tool_manager.register_tool("internet", fetch_summary)
        self.tool_manager.register_tool("file", FileTool())

    def register_tool(self, name, tool_func):
        self.tool_manager.register_tool(name, tool_func)

    def think(self, input_text):
        self.memory.add(input_text)
        return f"{self.name} is thinking about: {input_text}"

    def recall(self):
        return self.memory.get_all()

    def act(self, tool_name, *args, **kwargs):
        return self.tool_manager.use_tool(tool_name, *args, **kwargs)
