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


from core.agent import PrometheusAgent
from tools.calculator import CalculatorTool
from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.summarizer_tool import SummarizerTool
from tools.internet_tool import InternetTool  # FIXED: correct class name

# Initialize tools
tools = [
    CalculatorTool(),
    NoteTool(),
    FileTool(),
    SummarizerTool(),
    InternetTool(),  # FIXED: correct tool class
]

# Initialize Prometheus AI
agent = PrometheusAgent(tools=tools)

# === OPTIONAL: Set SerpAPI key directly in Colab ===
import os
os.environ["SERPAPI_API_KEY"] = "5f4c682efd58236a55d6a7de3fe8a792d933125c8157047a26e0e9c2a9cd5e37"

# === Run a test prompt ===
response = agent.run("Search the web for the latest updates on GPT-5.")
print(response)
