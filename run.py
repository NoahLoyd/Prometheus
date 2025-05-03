from core.agent import PrometheusAgent
from tools.calculator import CalculatorTool
from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.summarizer_tool import SummarizerTool
from tools.internet_tool import InternetTool  # <-- FIXED: correct class name

# Initialize tools
tools = [
    CalculatorTool(),
    NoteTool(),
    FileTool(),
    SummarizerTool(),
    InternetTool(),  # <-- FIXED: correct tool class
]

# Initialize Prometheus AI
agent = PrometheusAgent(tools=tools)

# === OPTIONAL: Set SerpAPI key directly in Colab ===
import os
os.environ["SERPAPI_API_KEY"] = "5f4c682efd58236a55d6a7de3fe8a792d933125c8157047a26e0e9c2a9cd5e37"

# === Run a test prompt ===
response = agent.run("Search the web for the latest updates on GPT-5.")
print(response)
