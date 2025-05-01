# tools/internet_tool.py

import requests
from bs4 import BeautifulSoup

def fetch_summary(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get page title and first paragraph
        title = soup.title.string if soup.title else "No title"
        paragraphs = soup.find_all('p')
        first_paragraph = paragraphs[0].get_text() if paragraphs else "No readable paragraph found."

        return f"{title}\n\n{first_paragraph.strip()}"
    except Exception as e:
        return f"Failed to fetch URL: {str(e)}"
