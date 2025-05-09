# run.py

import os
from core.agent import PrometheusAgent
from tools.calculator import CalculatorTool
from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.summarizer_tool import SummarizerTool
from tools.internet_tool import InternetTool

# === Load SerpAPI Key ===
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY", "5f4c682efd58236a55d6a7de3fe8a792d933125c8157047a26e0e9c2a9cd5e37")

# === Initialize Tools ===
tools = [
    CalculatorTool(),
    NoteTool(),
    FileTool(),
    SummarizerTool(),
    InternetTool(),
]

# === Initialize Agent ===
agent = PrometheusAgent(tools=tools)

# === Tool Tests ===
def run_tests():
    print("\n[Calculator]")
    print("2 + 2 =>", agent.run("calculator: 2 + 2"))
    print("Invalid =>", agent.run("calculator: invalid input"))

    print("\n[Notes]")
    print(agent.run("note: save: Remember to invest in GPUs"))
    print(agent.run("note: list"))

    print("\n[File Tool]")
    print(agent.run("file: write:test.txt:This is a test file."))
    print(agent.run("file: read:test.txt"))
    print(agent.run("file: list"))
    print(agent.run("file: read:nonexistent.txt"))

    print("\n[Summarizer]")
    print(agent.run("summarize: Artificial intelligence is a rapidly evolving field that will transform every industry."))

    print("\n[Internet Tool]")
    print(agent.run("internet: GPT-5 latest news"))

# === Run Tests ===
if __name__ == "__main__":
    run_tests()