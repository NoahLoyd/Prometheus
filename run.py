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
    InternetTool()
]

# Initialize Prometheus agent
agent = PrometheusAgent(tools=tools)

# === Test each tool ===
print(agent.run("calculator: 2 + 2"))
print(agent.run("note: save: Remember to invest in GPUs"))
print(agent.run("note: list"))
print(agent.run("file: list"))
print(agent.run("internet: Search the web for the latest updates on GPT-5"))
