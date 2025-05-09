# tools/internet_tool.py
import os
import requests
from tools.base_tool import BaseTool

class InternetTool(BaseTool):
    def __init__(self):
        super().__init__(name="internet", description="Perform internet searches using SERPAPI.")

    def run(self, query: str) -> str:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Error: SERPAPI_API_KEY not set."

        params = {"q": query, "api_key": api_key, "engine": "google"}
        try:
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            results = response.json().get("organic_results", [])
            if not results:
                return "No results found."
            return "\n\n".join(f"{r['title']}\n{r['link']}" for r in results[:3])
        except Exception as e:
            return f"Search failed: {e}"
