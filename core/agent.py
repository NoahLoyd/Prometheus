# core/agent.py

from tools.calculator import CalculatorTool
from tools.file_tool import FileTool
from tools.internet_tool import InternetTool
from tools.note_tool import NoteTool
from tools.summarizer_tool import SummarizerTool
from tools.tool_manager import ToolManager

class PrometheusAgent:
    def __init__(self, tools=None):
        self.tool_manager = ToolManager()

        # Instantiate tools
        calculator_tool = CalculatorTool()
        file_tool = FileTool()
        internet_tool = InternetTool()
        note_tool = NoteTool()
        summarizer_tool = SummarizerTool()

        # Register tools with the correct callable function
        self.tool_manager.register_tool(calculator_tool, calculator_tool.run)
        self.tool_manager.register_tool(file_tool, file_tool.run)
        self.tool_manager.register_tool(internet_tool, internet_tool.run)
        self.tool_manager.register_tool(note_tool, note_tool.save_note)
        self.tool_manager.register_tool(summarizer_tool, summarizer_tool.run)

        # Optional: support passing in additional tools
        if tools:
            for tool in tools:
                self.tool_manager.register_tool(tool, tool.run)

    # Add this so PrometheusAgent can run commands directly
    def run(self, command: str) -> str:
        return self.tool_manager.run_tool(command)
