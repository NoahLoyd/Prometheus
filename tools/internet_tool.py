# /tools/internet_tool.py

from tools.base_tool import BaseTool
import requests
import os

class InternetSearchTool(BaseTool):
    def __init__(self):
        super().__init__("internet_search", "Search the web using SerpAPI")

    def run(self, query):
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "SerpAPI key not found. Please set the SERPAPI_API_KEY environment variable."

        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "engine": "google"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "organic_results" in data and data["organic_results"]:
                return data["organic_results"][0].get("snippet", "No snippet found.")
            else:
                return "No results found."
        else:
            return f"Request failed with status code {response.status_code}"

# This line must be present for Prometheus to register the tool
tool = InternetSearchTool()
