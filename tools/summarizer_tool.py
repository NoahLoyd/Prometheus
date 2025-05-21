# tools/summarizer_tool.py

from tools.base_tool import BaseTool

class SummarizerTool(BaseTool):
    name = "summarizer"
    description = "Summarizes a block of text using simple extractive summarization."

    def run(self, query: str) -> str:
        try:
            # Simple extractive summary: return first and last sentence.
            sentences = [s.strip() for s in query.strip().split('.') if s.strip()]
            if not sentences:
                return "No content to summarize."
            if len(sentences) == 1:
                return sentences[0]
            return f"{sentences[0]}. ... {sentences[-1]}."
        except Exception as e:
            return f"SummarizerTool error: {e}"
