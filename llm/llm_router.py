from typing import List, Tuple, Dict, Optional
from .local_llm import LocalLLM
from .base_llm import BaseLLM
from core.logging import Logging
from concurrent.futures import ThreadPoolExecutor
import time


class LLMRouter:
    """
    An advanced router that intelligently selects, manages, and executes requests
    to different LLMs based on scoring metrics, adaptive feedback, and historical performance.
    Supports parallel planning, self-benchmarking, memory-driven routing, and self-debugging.
    """

    def __init__(self, config: Dict[str, Dict]):
        """
        Initialize the router with a configuration of models.

        :param config: Dictionary with model names as keys and their parameters as values.
                       Example: {"mistral": {"path": "mistral-7b", "type": "local", "tags": ["text-gen"]}}
        """
        self.config = config
        self.models = {}
        self.logger = Logging()
        self.feedback_weights = {}  # Dynamic task-specific feedback weights
        self.task_profiles = {}  # Tracks task-specific performance profiles for models

    def _get_model(self, name: str) -> BaseLLM:
        """
        Retrieve or lazily load a model based on its name.

        :param name: The name of the model to retrieve.
        :return: An instance of BaseLLM.
        :raises: Exception if all models fail during initialization.
        """
        if name not in self.models:
            try:
                model_config = self.config[name]
                if model_config["type"] == "local":
                    self.models[name] = LocalLLM(model_path=model_config["path"], device=model_config.get("device", "cpu"))
                # Add other model types here
            except Exception as e:
                self.logger.log_event("model_fail", f"Failed to load model {name}: {str(e)}")
                raise e

        return self.models[name]

    def generate_plan(self, goal: str, context: Optional[str] = None, task_type: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Generate a plan using the best available models in parallel and compare their outputs.

        :param goal: The goal or input to the model.
        :param context: Optional context for the model to use.
        :param task_type: Optional task type for relevance scoring.
        :return: A list of (tool_name, query) tuples representing the plan.
        """
        top_models = self._select_top_models(task_type, num_models=3)
        results = self._execute_in_parallel(top_models, goal, context)

        # Evaluate and select the best result
        best_result = self._evaluate_results(results, goal, task_type)
        self._update_model_profiles(best_result["model_name"], task_type, success=True)

        return best_result["plan"]

    def _select_top_models(self, task_type: Optional[str], num_models: int = 3) -> List[str]:
        """
        Select the top N models based on scoring.

        :param task_type: Optional task type for relevance scoring.
        :param num_models: Number of top models to select.
        :return: A list of model names.
        """
        scores = {}
        for model_name, model_config in self.config.items():
            success_rate = self.logger.get_model_success_rate(model_name)
            response_time = self.logger.get_model_avg_latency(model_name)
            relevance = 1 if task_type and task_type in model_config.get("tags", []) else 0
            feedback_weight = self.feedback_weights.get((model_name, task_type), 1.0)

            scores[model_name] = (
                0.4 * success_rate +
                0.3 * (1 / (response_time + 1e-6)) +
                0.2 * relevance +
                0.1 * feedback_weight
            )

        # Sort models by score and return the top N
        sorted_models = sorted(scores, key=scores.get, reverse=True)
        return sorted_models[:num_models]

    def _execute_in_parallel(self, models: List[str], goal: str, context: Optional[str]) -> List[Dict]:
        """
        Execute the goal in parallel across multiple models and gather results.

        :param models: List of model names to execute.
        :param goal: The goal or input for the models.
        :param context: Optional context for the models.
        :return: A list of result dictionaries from each model.
        """
        results = []

        def execute_model(model_name):
            try:
                start_time = time.time()
                model = self._get_model(model_name)
                plan = model.generate_plan(goal, context)
                elapsed_time = time.time() - start_time

                return {
                    "model_name": model_name,
                    "plan": plan,
                    "success": True,
                    "latency": elapsed_time
                }
            except Exception as e:
                return {"model_name": model_name, "plan": None, "success": False, "error": str(e)}

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(execute_model, model_name) for model_name in models]
            results = [future.result() for future in futures]

        return results

    def _evaluate_results(self, results: List[Dict], goal: str, task_type: Optional[str]) -> Dict:
        """
        Evaluate and compare results from multiple models to select the best one.

        :param results: List of result dictionaries from models.
        :param goal: The goal or input for evaluation.
        :param task_type: Optional task type for evaluation relevance.
        :return: The best result dictionary.
        """
        # Placeholder for more advanced evaluation logic
        # Currently selects the first successful result
        for result in results:
            if result["success"]:
                return result

        # If no successful results, raise an error
        raise RuntimeError("All models failed to generate a valid plan.")

    def _update_model_profiles(self, model_name: str, task_type: Optional[str], success: bool):
        """
        Update task-specific performance profiles for models.

        :param model_name: The name of the model.
        :param task_type: The type of task the model was used for.
        :param success: Whether the model succeeded.
        """
        key = (model_name, task_type)
        if key not in self.task_profiles:
            self.task_profiles[key] = {"success_count": 0, "failure_count": 0}

        if success:
            self.task_profiles[key]["success_count"] += 1
        else:
            self.task_profiles[key]["failure_count"] += 1

    def self_debug(self):
        """
        Analyze repeated failures and generate recommendations for routing or model updates.
        """
        failure_patterns = {}
        for key, profile in self.task_profiles.items():
            failure_ratio = profile["failure_count"] / (profile["success_count"] + profile["failure_count"])
            if failure_ratio > 0.5:
                failure_patterns[key] = failure_ratio

        # Log recommendations
        for (model_name, task_type), ratio in failure_patterns.items():
            self.logger.log_event(
                tag="self_debug",
                detail=f"Model '{model_name}' has high failure ratio ({ratio:.2f}) for task type '{task_type}'."
            )