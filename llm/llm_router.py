from typing import List, Tuple, Dict, Optional
from .evaluation_strategy import EvaluationStrategy
from .fallback_strategy import FallbackStrategy
from .plan_voting_strategy import PlanVotingStrategy
from .local_llm import LocalLLM
from .base_llm import BaseLLM
from core.logging import Logging
from concurrent.futures import ThreadPoolExecutor
import time


class LLMRouter:
    def __init__(self, config: Dict[str, Dict], evaluation_strategy: EvaluationStrategy,
                 fallback_strategy: FallbackStrategy, voting_strategy: PlanVotingStrategy):
        self.config = config
        self.models: Dict[str, BaseLLM] = {}
        self.logger = Logging()
        self.feedback_weights: Dict[Tuple[str, Optional[str]], float] = {}
        self.task_profiles: Dict[Tuple[str, Optional[str]], Dict[str, int]] = {}

        self.evaluation_strategy = evaluation_strategy
        self.fallback_strategy = fallback_strategy
        self.voting_strategy = voting_strategy

    def _get_model(self, name: str) -> BaseLLM:
        if name not in self.models:
            model_config = self.config.get(name)
            if not model_config:
                raise ValueError(f"Model '{name}' is not configured.")
            if model_config["type"] == "local":
                self.models[name] = LocalLLM(model_path=model_config["path"], device=model_config.get("device", "cpu"))
        return self.models[name]

    def generate_plan(self, goal: str, context: Optional[str] = None, task_type: Optional[str] = None) -> List[Tuple[str, str]]:
        top_models = self._select_top_models(task_type, num_models=3)
        results = self._execute_in_parallel(top_models, goal, context)

        if not any(result["success"] for result in results):
            return self.fallback_strategy.refine_plan(goal, context, task_type)

        if sum(1 for result in results if result["success"]) > 1:
            return self.voting_strategy.merge_or_vote(results)

        best_result = self.evaluation_strategy.evaluate(results, goal, task_type)
        self._update_model_profiles(best_result["model_name"], task_type, success=True)
        return best_result["plan"]

    # Additional methods (_select_top_models, _execute_in_parallel, _update_model_profiles) remain unchanged.