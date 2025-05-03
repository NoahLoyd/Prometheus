from base_tool import BaseTool

class SummarizerTool(BaseTool):
    def __init__(self):
        super().__init__(name="summarizer", description="Summarize a long text into a shorter version with key points.")

    def use(self, input_text: str) -> str:
        if not input_text.strip():
            return "Please provide some text to summarize."

        # Simulated summarization for now — replace with real LLM logic if needed
        import textwrap
        summary = (
            "Summary:\n"
            + "\n".join(textwrap.wrap(input_text.strip()[:500], width=80))  # Trims & wraps
        )
        return summary
