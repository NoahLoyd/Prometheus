
import os
import requests
from tools.base_tool import BaseTool

class InternetSearchTool(BaseTool):
    def __init__(self):
        super().__init__(name="internet_search", description="Searches the internet using SerpAPI.")

    def run(self, query: str) -> str:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Error: SERPAPI_API_KEY not found in environment variables."

        params = {
            "q": query,
            "api_key": api_key,
            "engine": "google"
        }

        try:
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
        except requests.RequestException as e:
            return f"Error: Failed to fetch results – {str(e)}"

        data = response.json()
        results = data.get("organic_results", [])
        if not results:
            return "No search results found."

        return "\n\n".join([f"{r.get('title')}\n{r.get('link')}" for r in results])

tool = InternetSearchTool()
