"""
Base tool interface - referenced but may be incomplete.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTool(ABC):
    """Base class for all Promethyn tools."""
    
    def __init__(self):
        self.name = self.__class__.__name__.lower().replace('tool', '')
    
    @abstractmethod
    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the tool with given query.
        
        :param query: Input query string
        :return: Result dictionary with success/error status
        """
        pass
    
    def test(self) -> Dict[str, Any]:
        """
        Run self-test to validate tool functionality.
        
        :return: Test result dictionary
        """
        try:
            result = self.run("test")
            return {
                "success": True,
                "result": result,
                "info": f"{self.name} self-test passed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"{self.name} self-test failed: {str(e)}"
            }
