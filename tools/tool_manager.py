# tool_manager.py
from tools.calculator import tool as CalculatorTool
from tools.file_tool import tool as FileTool
from tools.notepad import tool as NoteTool
from tools.internet_tool import tool as InternetTool
from tools.search_tool import tool as SearchTool
from tools.summarizer_tool import tool as SummarizerTool  # <- make sure this file exists
from tools.summarize_web_tool import tool as SummarizeWebTool  # <- new web summary tool

class ToolManager:
    def __init__(self):
        self.tools = [
            CalculatorTool(),
            FileTool(),
            NoteTool(),
            InternetTool(),
            SearchTool(),
            SummarizerTool(),
            SummarizeWebTool(),  # <- newly added
        ]

    def get_tool(self, name):
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def list_tools(self):
        return [tool.name for tool in self.tools]
