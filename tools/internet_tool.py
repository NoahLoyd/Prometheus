import os
from serpapi import GoogleSearch

class SearchTool:
    def __init__(self):
        self.api_key = "5f4c682efd58236a55d6a7de3fe8a792d933125c8157047a26e0e9c2a9cd5e37"

    def run(self, query):
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            return f"Search failed: {results['error']}"

        if "organic_results" not in results:
            return "No results found."

        output = []
        for result in results["organic_results"][:5]:
            title = result.get("title", "No title")
            link = result.get("link", "No link")
            snippet = result.get("snippet", "")
            output.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n")

        return "\n".join(output)
