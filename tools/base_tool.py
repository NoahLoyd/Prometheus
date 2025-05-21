# tools/base_tool.py

from abc import ABC, abstractmethod

class BaseTool(ABC):
    """
    Abstract base class for all tools in Promethyn.
    Each tool must have a unique string 'name' and implement run(query: str) -> str.
    """

    name: str

    @abstractmethod
    def run(self, query: str) -> str:
        """
        Processes the given query string and returns a string result.
        """
        pass
