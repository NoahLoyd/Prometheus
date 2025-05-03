import os
import requests
from tools.base_tool import BaseTool

class InternetSearchTool(BaseTool):
    def __init__(self):
        super().__init__("internet_search", "Search the web using SerpAPI")

    def use(self, query: str) -> str:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Error: SERPAPI_API_KEY is not set. Please set it using os.environ or a .env file."

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": 5,
        }

        try:
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
        except requests.RequestException as e:
            return f"Error: Failed to fetch results — {str(e)}"

        data = response.json()
        results = data.get("organic_results", [])
        if not results:
            return "No search results found."

        return "\n\n".join([f"{r.get('title')}\n{r.get('link')}" for r in results])

         tool = InternetSearchTool()
