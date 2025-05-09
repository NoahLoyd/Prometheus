import os
from core.agent import PrometheusAgent
from tools.calculator import CalculatorTool
from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.summarizer_tool import SummarizerTool
from tools.internet_tool import InternetTool
from tools.tool_manager import ToolManager  # Fixed import for ToolManager

# Securely load the SERPAPI API key
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY", "default_key_here")

# Initialize tools
tools = [
    CalculatorTool(),
    NoteTool(),
    FileTool(),
    SummarizerTool(),
    InternetTool(),
]

# Initialize ToolManager and register tools
tool_manager = ToolManager()
for tool in tools:
    tool_manager.register_tool(tool.name, tool)

# Initialize agent with ToolManager
agent = PrometheusAgent(tools=tool_manager)

# === Run Tests ===
def run_tests():
    try:
        # Calculator Tests
        print("Calculator (2 + 2):", agent.run("calculator: 2 + 2"))
        print("Calculator (Invalid Input):", agent.run("calculator: invalid input"))
    except Exception as e:
        print(f"Calculator Error: {e}")

    try:
        # Note Tool Tests
        print("Note Save:", agent.run("note: save: Remember to invest in GPUs"))
        print("Note List:", agent.run("note: list"))
    except Exception as e:
        print(f"Note Tool Error: {e}")

    try:
        # File Tool Tests
        print("File Write:", agent.run("file: write:test.txt:This is a test file."))
        print("File Read:", agent.run("file: read:test.txt"))
        print("File List:", agent.run("file: list"))
        print("File Read Nonexistent:", agent.run("file: read:nonexistent.txt"))
    except Exception as e:
        print(f"File Tool Error: {e}")

    try:
        # Summarizer Tool Tests
        print(
            "Summarizer:",
            agent.run(
                "summarize: Artificial intelligence is a rapidly evolving field that..."
            ),
        )
    except Exception as e:
        print(f"Summarizer Tool Error: {e}")

    try:
        # Internet Tool Tests
        print("Internet (GPT-5 Latest News):", agent.run("internet: GPT-5 latest news"))
    except Exception as e:
        print(f"Internet Tool Error: {e}")


# Run all tests
if __name__ == "__main__":
    run_tests()