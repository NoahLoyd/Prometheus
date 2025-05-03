# summarize_web_tool.py
from tools.base_tool import BaseTool
from tools.internet_tool import tool as internet_tool

class SummarizeWebTool(BaseTool):
    def run(self, input_text):
        try:
            # Step 1: Search the web
            search_results = search_tool.run(input_text)
            links = [entry['link'] for entry in search_results[:3]]  # top 3 links

            summaries = []
            for link in links:
                content = internet_tool.run(link)
                if not content:
                    continue

                # Extract simple summary (basic filter, could later use LLMs)
                lines = content.splitlines()
                important_lines = [
                    line.strip() for line in lines 
                    if len(line.strip()) > 50 and '.' in line
                ]
                summary = '\n'.join(important_lines[:3])  # get 3 important lines
                summaries.append(f"From {link}:\n{summary}")

            return '\n\n'.join(summaries) if summaries else "No relevant content found."
        except Exception as e:
            return f"Error summarizing from the web: {e}"

tool = SummarizeWebTool()
