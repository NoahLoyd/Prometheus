from typing import List, Tuple, Dict, Optional
from .local_llm import LocalLLM
from .base_llm import BaseLLM
from core.logging import Logging
import time

class LLMRouter:
    """
    A router that intelligently selects, manages, and executes requests
    to different LLMs based on scoring metrics.
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

    def _get_model(self, name: str) -> BaseLLM:
        """
        Retrieve or lazily load a model based on its name.

        :param name: The name of the model to retrieve.
        :return: An instance of BaseLLM.
        :raises: Exception if all models fail during initialization.
        """
        if name not in self.models:
            try:
                # Lazy load model based on type
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
        Generate a plan using the best available model.

        :param goal: The goal or input to the model.
        :param context: Optional context for the model to use.
        :param task_type: Optional task type for relevance scoring.
        :return: A list of (tool_name, query) tuples representing the plan.
        """
        best_model_name = self._select_best_model(task_type)
        try:
            start_time = time.time()
            model = self._get_model(best_model_name)
            result = model.generate_plan(goal, context)
            elapsed_time = time.time() - start_time

            # Log success
            self.logger.log_model_performance(best_model_name, task_type, success=True, latency=elapsed_time)

            return result
        except Exception as e:
            # Log failure and attempt failover
            self.logger.log_model_performance(best_model_name, task_type, success=False, latency=None)
            failover_model_name = self._failover_best_model(best_model_name)
            if failover_model_name:
                return self.generate_plan(goal, context, task_type)
            else:
                raise RuntimeError(f"All models failed to generate a plan: {str(e)}")

    def _select_best_model(self, task_type: Optional[str] = None) -> str:
        """
        Select the best model based on scoring.

        :param task_type: Optional task type for relevance scoring.
        :return: The name of the best model.
        """
        scores = {}
        for model_name, model_config in self.config.items():
            # Use historical success rate and response time
            success_rate = self.logger.get_model_success_rate(model_name)
            response_time = self.logger.get_model_avg_latency(model_name)
            relevance = 1 if task_type and task_type in model_config.get("tags", []) else 0

            # Calculate a weighted score
            scores[model_name] = (
                0.6 * success_rate +
                0.3 * (1 / (response_time + 1e-6)) +  # Inverse latency
                0.1 * relevance
            )

        # Prioritize local open-source models
        open_source_models = [name for name, config in self.config.items() if config["type"] == "local"]
        best_model = max(scores, key=scores.get)
        if best_model not in open_source_models:
            fallback = max(open_source_models, key=lambda name: scores.get(name, 0))
            return fallback if fallback in scores else best_model

        return best_model

    def _failover_best_model(self, failed_model_name: str) -> Optional[str]:
        """
        Attempt to find the next-best model for failover.

        :param failed_model_name: The name of the model that failed.
        :return: The name of the next-best model, or None if no alternatives exist.
        """
        available_models = [name for name in self.config if name != failed_model_name]
        if not available_models:
            return None

        # Recalculate scores without the failed model
        task_type = None  # Optionally pass task_type for more specific failovers
        scores = {name: self._select_best_model(task_type) for name in available_models}
        next_best_model = max(scores, key=scores.get, default=None)

        return next_best_model

    def log_model_performance(self, model_name: str, task_type: Optional[str], success: bool, latency: Optional[float]):
        """
        Log the performance of a model.

        :param model_name: The name of the model.
        :param task_type: The type of task the model was used for.
        :param success: Whether the model succeeded.
        :param latency: The time taken for the model to complete the task, in seconds.
        """
        self.logger.log_tool_performance(
            tool_name=model_name,
            success=success
        )
        self.logger.log_event(
            tag="model_performance",
            detail={
                "model_name": model_name,
                "task_type": task_type,
                "success": success,
                "latency": latency,
            }
        )