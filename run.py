# run.py
from core.agent import PrometheusAgent
from tools.calculator import CalculatorTool
from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.summarizer_tool import SummarizerTool
from tools.internet_tool import InternetTool

import os
os.environ["SERPAPI_API_KEY"] = "5f4c682efd58236a55d6a7de3fe8a792d933125c8157047a26e0e9c2a9cd5e37"

tools = [
    CalculatorTool(),
    NoteTool(),
    FileTool(),
    SummarizerTool(),
    InternetTool()
]

agent = PrometheusAgent(tools=tools)

# === Run tool tests ===
print(agent.run("calculator: 2 + 2"))
print(agent.run("note: Remember to invest in GPUs"))
print(agent.run("file: list"))
print(agent.run("internet: GPT-5 news"))
