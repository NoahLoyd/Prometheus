from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging

# Attempt to import all required modules; provide safe fallbacks if needed
try:
    from llm.evaluation_strategy import EvaluationStrategy
except ImportError as e:
    logging.warning("Failed to import EvaluationStrategy: %s", e)
    EvaluationStrategy = None

try:
    from llm.fallback_strategy import FallbackStrategy
except ImportError as e:
    logging.warning("Failed to import FallbackStrategy: %s", e)
    FallbackStrategy = None

try:
    from llm.voting_strategy import VotingStrategy
except ImportError as e:
    logging.warning("Failed to import VotingStrategy: %s", e)
    VotingStrategy = None

try:
    from llm.local_llm import LocalLLM
except ImportError as e:
    logging.warning("Failed to import LocalLLM: %s", e)
    LocalLLM = None

try:
    from llm.base_llm import BaseLLM
except ImportError as e:
    logging.warning("Failed to import BaseLLM: %s", e)
    BaseLLM = None

try:
    from llm.task_profiler import TaskProfiler
except ImportError as e:
    logging.warning("Failed to import TaskProfiler: %s", e)
    TaskProfiler = None

try:
    from llm.confidence_scorer import ConfidenceScorer
except ImportError as e:
    logging.warning("Failed to import ConfidenceScorer: %s", e)
    ConfidenceScorer = None

try:
    from llm.feedback_memory import FeedbackMemory
except ImportError as e:
    logging.warning("Failed to import FeedbackMemory: %s", e)
    FeedbackMemory = None

try:
    from core.logging import Logging
except ImportError as e:
    logging.warning("Failed to import Logging: %s", e)
    Logging = logging.getLogger(__name__)  # Fallback to default Python logging if Core Logging is unavailable


class LLMRouter:
    def __init__(self, config: Dict[str, Dict], evaluation_strategy: Optional[EvaluationStrategy] = None,
                 fallback_strategy: Optional[FallbackStrategy] = None,
                 voting_strategy: Optional[VotingStrategy] = None,
                 profiler: Optional[TaskProfiler] = None,
                 feedback_memory: Optional[FeedbackMemory] = None,
                 confidence_scorer: Optional[ConfidenceScorer] = None):
        """
        Initialize the LLM Router with configurations and strategies.

        :param config: Configuration for models (paths, types, devices).
        :param evaluation_strategy: Strategy for model performance evaluation.
        :param fallback_strategy: Strategy for fallback in case of failures.
        :param voting_strategy: Strategy for merging or voting on model outputs.
        :param profiler: Task Profiler for task classification.
        :param feedback_memory: Feedback Memory system for storing task results.
        :param confidence_scorer: Confidence Scorer for evaluating outputs.
        """
        self.config = config
        self.models: Dict[str, BaseLLM] = {}
        self.logger = Logging()
        self.task_profiles: Dict[Tuple[str, Optional[str]], Dict[str, int]] = {}
        self.cache: Dict[str, List[Tuple[str, str]]] = {}
        self.evaluation_strategy = evaluation_strategy or EvaluationStrategy()
        self.fallback_strategy = fallback_strategy or FallbackStrategy()
        self.voting_strategy = voting_strategy or VotingStrategy()
        self.profiler = profiler or TaskProfiler()
        self.feedback_memory = feedback_memory or FeedbackMemory()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()

    def _get_model(self, name: str) -> BaseLLM:
        """Retrieve or initialize a model by name."""
        if name not in self.models:
            model_config = self.config.get(name)
            if not model_config:
                raise ValueError(f"Model '{name}' is not configured.")
            if model_config["type"] == "local":
                if not LocalLLM:
                    raise ImportError("LocalLLM is not available.")
                self.models[name] = LocalLLM(
                    model_path=model_config["path"],
                    device=model_config.get("device", "cpu")
                )
        return self.models[name]

    def _hash_query(self, goal: str, context: Optional[str], task_type: Optional[str]) -> str:
        """Generate a unique hash for caching based on the query."""
        query_str = f"{goal}:{context}:{task_type}"
        return hashlib.sha256(query_str.encode()).hexdigest()

    def generate_plan(self, goal: str, context: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Generate a task execution plan based on the goal and context.

        :param goal: The primary task or objective.
        :param context: Additional context for the task.
        :return: A list of task execution plans.
        """
        self.logger.info(f"Generating plan for goal: {goal}")

        # Classify task type using Task Profiler
        task_type = self.profiler.classify_task(goal) if self.profiler else None
        self.logger.info(f"Task type classified as: {task_type}")

        # Check cache
        query_hash = self._hash_query(goal, context, task_type)
        if query_hash in self.cache:
            self.logger.info("Cache hit for query.")
            return self.cache[query_hash]

        # Select models
        top_models = self._select_top_models(task_type, num_models=3)
        results = self._execute_in_parallel(top_models, goal, context)

        # Evaluate confidence and apply fallback if necessary
        confidences = [self.confidence_scorer.compute_confidence(result) for result in results if self.confidence_scorer]
        if max(confidences, default=0) < 0.5:  # Threshold for fallback
            self.logger.warning("Confidence below threshold. Applying fallback strategy.")
            refined_plan = self.fallback_strategy.refine_plan(goal, context, task_type) if self.fallback_strategy else []
            if self.feedback_memory:
                self.feedback_memory.record_feedback(task_type, "fallback", success=True, confidence=0.5)
            return refined_plan

        # Merge or vote if multiple models succeed
        successful_results = [results[i] for i in range(len(results)) if results[i].get("success")]
        if len(successful_results) > 1 and self.voting_strategy:
            merged_plan = self.voting_strategy.merge_or_vote(successful_results)
            if self.feedback_memory:
                self.feedback_memory.record_feedback(task_type, "voted", success=True, confidence=max(confidences))
            self.cache[query_hash] = merged_plan
            return merged_plan

        # Evaluate and select the best result
        if self.evaluation_strategy:
            best_result = self.evaluation_strategy.evaluate(results, goal, task_type)
            self._update_model_profiles(best_result["model_name"], task_type, success=True)
            if self.feedback_memory:
                self.feedback_memory.record_feedback(
                    task_type,
                    best_result["model_name"],
                    success=True,
                    confidence=self.confidence_scorer.compute_confidence(best_result) if self.confidence_scorer else 0.0
                )
            self.cache[query_hash] = best_result["plan"]
            return best_result["plan"]

        return []

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
                executor.submit(self._get_model(model).generate_plan, goal, context): model
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