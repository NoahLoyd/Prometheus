from typing import List, Tuple, Dict, Optional
from .evaluation_strategy import EvaluationStrategy
from .fallback_strategy import FallbackStrategy
from .voting_strategy import VotingStrategy
from .local_llm import LocalLLM
from .base_llm import BaseLLM
from core.logging import Logging
from concurrent.futures import ThreadPoolExecutor
import time


class LLMRouter:
    def __init__(self, config: Dict[str, Dict], evaluation_strategy: EvaluationStrategy,
                 fallback_strategy: FallbackStrategy, voting_strategy: VotingStrategy):
        """
        Initialize the LLM Router with configurations and strategies.

        :param config: Configuration for models (paths, types, devices).
        :param evaluation_strategy: Strategy for model performance evaluation.
        :param fallback_strategy: Strategy for fallback in case of failures.
        :param voting_strategy: Strategy for merging or voting on model outputs.
        """
        self.config = config
        self.models: Dict[str, BaseLLM] = {}
        self.logger = Logging()
        self.feedback_weights: Dict[Tuple[str, Optional[str]], float] = {}
        self.task_profiles: Dict[Tuple[str, Optional[str]], Dict[str, int]] = {}
        self.evaluation_strategy = evaluation_strategy
        self.fallback_strategy = fallback_strategy
        self.voting_strategy = voting_strategy

    def _get_model(self, name: str) -> BaseLLM:
        """Retrieve or initialize a model by name."""
        if name not in self.models:
            model_config = self.config.get(name)
            if not model_config:
                raise ValueError(f"Model '{name}' is not configured.")
            if model_config["type"] == "local":
                self.models[name] = LocalLLM(
                    model_path=model_config["path"],
                    device=model_config.get("device", "cpu")
                )
        return self.models[name]

    def generate_plan(self, goal: str, context: Optional[str] = None, task_type: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Generate a task execution plan based on the goal, context, and task type.

        :param goal: The primary task or objective.
        :param context: Additional context for the task.
        :param task_type: The type of task (e.g., 'code', 'math').
        :return: A list of task execution plans.
        """
        self.logger.info(f"Generating plan for goal: {goal} with task type: {task_type}")
        top_models = self._select_top_models(task_type, num_models=3)
        results = self._execute_in_parallel(top_models, goal, context)

        if not any(result["success"] for result in results):
            return self.fallback_strategy.refine_plan(goal, context, task_type)

        if sum(1 for result in results if result["success"]) > 1:
            return self.voting_strategy.merge_or_vote(results)

        best_result = self.evaluation_strategy.evaluate(results, goal, task_type)
        self._update_model_profiles(best_result["model_name"], task_type, success=True)
        return best_result["plan"]

    def _select_top_models(self, task_type: Optional[str], num_models: int) -> List[str]:
        """Select the top-performing models for a given task type."""
        sorted_models = sorted(
            self.task_profiles.items(),
            key=lambda x: x[1].get(task_type, 0),
            reverse=True
        )
        return [model[0][0] for model in sorted_models[:num_models]]

    def _execute_in_parallel(self, models: List[str], goal: str, context: Optional[str]) -> List[Dict]:
        """Execute tasks concurrently across multiple models."""
        results = []
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            future_to_model = {
                executor.submit(self._get_model(model).execute, goal, context): model
                for model in models
            }
            for future in future_to_model:
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=30)
                    result["model_name"] = model_name
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Model {model_name} failed with error: {e}")
                    results.append({"model_name": model_name, "success": False, "error": str(e)})
        return results

    def _update_model_profiles(self, model_name: str, task_type: Optional[str], success: bool):
        """Update model performance profiles based on task outcomes."""
        key = (model_name, task_type)
        if key not in self.task_profiles:
            self.task_profiles[key] = {"successes": 0, "failures": 0}
        if success:
            self.task_profiles[key]["successes"] += 1
        else:
            self.task_profiles[key]["failures"] += 1