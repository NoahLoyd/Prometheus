from tools.file_tool import FileTool
from tools.internet_tool import InternetSearchTool
from tools.calculator import CalculatorTool
from tools.note_tool import NoteTool
from tools.summarizer_tool import SummarizerTool
from tools.tool_manager import ToolManager
from memory.short_term import ShortTermMemory

class PrometheusAgent:
    def __init__(self, tools=None):
        # Initialize memory and tool manager
        self.memory = ShortTermMemory()
        self.tool_manager = ToolManager()

        # Register tools
        default_tools = [
            CalculatorTool(),
            FileTool(),
            NoteTool(),
            SummarizerTool(),
            InternetSearchTool()
        ]
        for tool in tools or default_tools:
            self.tool_manager.register_tool(tool)

    def run(self, command: str) -> str:
        self.memory.store("last_command", command)
        response = self.tool_manager.handle(command)
        self.memory.store("last_response", response)
        return response
