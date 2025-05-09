from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

class BaseLLM(ABC):
    @abstractmethod
    def generate_plan(self, goal: str, context: Optional[str] = None) -> List[Tuple[str, str]]:
        pass

    @abstractmethod
    def _format_prompt(self, goal: str, context: Optional[str]) -> str:
        pass

    @abstractmethod
    def _parse_plan(self, response: str) -> List[Tuple[str, str]]:
        pass