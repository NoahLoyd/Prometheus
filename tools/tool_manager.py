from tools.calculator import CalculatorTool
from tools.notepad import NotepadTool
from tools.file_tool import FileTool
# from tools.internet_tool import InternetSearchTool  # Enable after Step 3

class ToolManager:
    def __init__(self):
        self.tools = {
            "calculator": CalculatorTool(),
            "notepad": NotepadTool(),
            "file": FileTool(),
            # "internet": InternetSearchTool(),  # Enable after building it
        }

    def get_tool(self, name):
        return self.tools.get(name)

    def list_tools(self):
        return list(self.tools.keys())
