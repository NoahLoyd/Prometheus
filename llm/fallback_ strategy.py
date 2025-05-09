from typing import List, Tuple, Optional, Dict
from core.logging import Logging
from .base_llm import BaseLLM


class FallbackStrategy:
    def refine_plan(self, goal: str, context: Optional[str], task_type: Optional[str]) -> List[Tuple[str, str]]:
        ...


class ChainOfThoughtFallbackStrategy(FallbackStrategy):
    def __init__(self, logger: Logging, models: Dict[str, BaseLLM]):
        self.logger = logger
        self.models = models

    def refine_plan(self, goal: str, context: Optional[str], task_type: Optional[str]) -> List[Tuple[str, str]]:
        try:
            meta_model = self.models.get("fallback_meta_model")
            if not meta_model:
                raise RuntimeError("Fallback meta-model not configured.")
            memory_context = self.logger.retrieve_long_term_memory(goal, task_type)
            chain_of_thought = f"Why did this fail?\nWhat steps are missing?\n{memory_context}"
            refined_plan = meta_model.generate_plan(goal, f"{context}\n{chain_of_thought}")
            return refined_plan
        except Exception as e:
            self.logger.log_event("fallback_fail", f"Meta-model failed: {str(e)}")
            raise RuntimeError(f"Fallback strategy failed: {str(e)}")