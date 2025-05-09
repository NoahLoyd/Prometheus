# run.py

import os
from core.agent import PrometheusAgent
from tools.calculator import CalculatorTool
from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.summarizer_tool import SummarizerTool
from tools.internet_tool import InternetTool

# === Load SERPAPI Key ===
os.environ["SERPAPI_API_KEY"] = "5f4c682efd58236a55d6a7de3fe8a792d933125c8157047a26e0e9c2a9cd5e37"

# === Initialize Tools ===
tools = [
    CalculatorTool(),
    NoteTool(),
    FileTool(),
    SummarizerTool(),
    InternetTool()
]

# === Initialize Prometheus Agent ===
agent = PrometheusAgent(tools=tools)

# === Run Tests ===
def run_tests():
    print("Calculator Tool:")
    print(agent.run("calculator: 2 + 2"))
    print(agent.run("calculator: invalid * 5"))

    print("\nNote Tool:")
    print(agent.run("note: save: Remember to invest in GPUs"))
    print(agent.run("note: list"))

    print("\nFile Tool:")
    print(agent.run("file: write: test.txt: Hello from Prometheus"))
    print(agent.run("file: read: test.txt"))
    print(agent.run("file: list"))

    print("\nSummarizer Tool:")
    print(agent.run("summarizer: Prometheus is a modular AI framework that aims to surpass GPT-4 by integrating reasoning, memory, and tool usage."))

    print("\nInternet Tool:")
    print(agent.run("internet: latest news on GPT-5"))

if __name__ == "__main__":
    run_tests()