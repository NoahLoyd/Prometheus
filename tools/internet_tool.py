import os
import requests
from tools.base_tool import BaseTool

class InternetSearchTool(BaseTool):
    def __init__(self):
        super().__init__("internet_search", "Search the web using SerpAPI.")
        self.api_key = os.getenv("SERPAPI_API_KEY")  # You can override this in Colab if needed

    def run(self, query: str) -> str:
        if not self.api_key:
            return "SerpAPI key not found. Please set SERPAPI_API_KEY."

        print(f"[InternetSearchTool] Searching: {query}")
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": 5
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            return f"Search failed: {response.text}"

        results = response.json().get("organic_results", [])
        if not results:
            return "No search results found."

        output = ""
        for i, result in enumerate(results, 1):
            title = result.get("title")
            link = result.get("link")
            snippet = result.get("snippet", "")
            output += f"{i}. {title}\n{snippet}\n{link}\n\n"

        return output.strip()
