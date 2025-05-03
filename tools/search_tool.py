import os
import requests
from tools.base import BaseTool

class SearchTool(BaseTool):
    name = "search"
    description = "Performs a web search using SerpAPI. Use it to look up current or real-time information from the internet."

    def __init__(self):
        self.api_key = "5f4c682efd58236a55d6a7de3fe8a792d933125c8157047a26e0e9c2a9cd5e37"
        if not self.api_key:
            raise ValueError("SerpAPI key is missing.")

    def run(self, query):
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google"
        }

        try:
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("organic_results", [])
            if not results:
                return "No search results found."
            top_results = [f"{res.get('title')} - {res.get('link')}" for res in results[:3]]
            return "\n".join(top_results)
        except Exception as e:
            return f"Search failed: {str(e)}"
