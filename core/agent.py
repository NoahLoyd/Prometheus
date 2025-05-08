# core/agent.py

from tools.tool_manager import ToolManager
from tools.calculator import CalculatorTool
from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.summarizer_tool import SummarizerTool
from tools.internet_tool import InternetTool

class PrometheusAgent:
    def __init__(self, tools=None):
        self.tool_manager = ToolManager()

        # Register core tools with explicit names
        self.register_named_tool("calculator", CalculatorTool())
        self.register_named_tool("note", NoteTool())
        self.register_named_tool("file", FileTool())
        self.register_named_tool("summarize", SummarizerTool())
        self.register_named_tool("internet", InternetTool())

        # Allow optional tools passed in
        if tools:
            for tool in tools:
                self.register_named_tool(tool.__class__.__name__.lower(), tool)

    def register_named_tool(self, name, tool_instance):
        if hasattr(tool_instance, "run"):
            self.tool_manager.register_tool(name, tool_instance.run)
        else:
            self.tool_manager.register_tool(name, tool_instance)

    def run(self, command: str) -> str:
        """Handle commands like 'tool_name: input'."""
        if ":" in command:
            tool_name, query = command.split(":", 1)
            return self.tool_manager.call_tool(tool_name.strip().lower(), query.strip())
        else:
            return self.tool_manager.run_tool(command.strip())
