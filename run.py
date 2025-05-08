# run.py

from core.agent import PrometheusAgent
from tools.calculator import CalculatorTool
from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.summarizer_tool import SummarizerTool
from tools.internet_tool import InternetTool

import os
os.environ["SERPAPI_API_KEY"] = "5f4c682efd58236a55d6a7de3fe8a792d933125c8157047a26e0e9c2a9cd5e37"

# Initialize tools
tools = [
    CalculatorTool(),
    NoteTool(),
    FileTool(),
    SummarizerTool(),
    InternetTool(),
]

# Initialize agent
agent = PrometheusAgent(tools=tools)

# === Run Tests ===
print("Calculator:", agent.run("calculator: 2 + 2"))
print("Note Save:", agent.run("note: save: Remember to invest in GPUs"))
print("Note List:", agent.run("note: list"))
print("File Write:", agent.run("file: write:test.txt|This is a test file."))
print("File Read:", agent.run("file: read:test.txt"))
print("File List:", agent.run("file: list"))
print("Summarizer:", agent.run("summarize: Artificial intelligence is a rapidly evolving field that..."))
print("Internet:", agent.run("internet: GPT-5 latest news"))