from base_tool import BaseTool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core import PrometheusAgent

class SummarizerTool(BaseTool):
    def __init__(self):
        super().__init__(name="summarizer", description="Summarize long text using the LLM.")

    def use(self, input_text: str, agent: "PrometheusAgent" = None) -> str:
        if not input_text.strip():
            return "Please provide some text to summarize."

        prompt = f"Summarize the following text clearly and concisely:\n\n{input_text}"
        response = agent.llm.generate(prompt)
        return response
