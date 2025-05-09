# tools/summarizer_tool.py
from tools.base_tool import BaseTool
from textwrap import shorten

class SummarizerTool(BaseTool):
    def __init__(self):
        super().__init__(name="summarizer", description="Summarizes long text into a short version.")

    def run(self, query: str) -> str:
        return shorten(query, width=100, placeholder="...")
