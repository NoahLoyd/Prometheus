from tools.file_tool import FileTool
from tools.internet_tool import InternetSearchTool
from tools.calculator import CalculatorTool
from tools.note_tool import NoteTool
from tools.summarizer_tool import SummarizerTool
from tools.tool_manager import ToolManager
from memory.short_term import ShortTermMemory

class PrometheusAgent:
    def __init__(self):
        # Initialize tools
        self.memory = ShortTermMemory()
        self.tool_manager = ToolManager()

        # Register tools
        self.tool_manager.register_tool(CalculatorTool())
        self.tool_manager.register_tool(FileTool())
        self.tool_manager.register_tool(NoteTool())
        self.tool_manager.register_tool(SummarizerTool())
        self.tool_manager.register_tool(InternetSearchTool())

    def run(self, command: str) -> str:
        self.memory.store("last_command", command)
        response = self.tool_manager.handle(command)
        self.memory.store("last_response", response)
        return response
