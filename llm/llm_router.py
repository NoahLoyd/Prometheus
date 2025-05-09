from typing import List, Tuple, Dict, Optional, Protocol
from .local_llm import LocalLLM
from .base_llm import BaseLLM
from core.logging import Logging
from concurrent.futures import ThreadPoolExecutor
import time


# Protocols for Strategy Interfaces
class EvaluationStrategy(Protocol):
    def evaluate(self, results: List[Dict], goal: str, task_type: Optional[str]) -> Dict:
        ...


class FallbackStrategy(Protocol):
    def refine_plan(self, goal: str, context: Optional[str], task_type: Optional[str]) -> List[Tuple[str, str]]:
        ...


class PlanVotingStrategy(Protocol):
    def merge_or_vote(self, results: List[Dict]) -> List[Tuple[str, str]]:
        ...


# Implementation of Evaluation Strategy
class DefaultEvaluationStrategy:
    def __init__(self, logger: Logging):
        self.logger = logger

    def evaluate(self, results: List[Dict], goal: str, task_type: Optional[str]) -> Dict:
        """
        Evaluate and compare results from multiple models to select the best one.
        Adds semantic similarity and entropy-based diversity scoring.
        """
        def score_result(result: Dict) -> float:
            if not result["success"]:
                return 0.0

            plan = result["plan"]
            # Semantic similarity scoring (mocked for demonstration)
            semantic_similarity = self.logger.calculate_semantic_similarity(goal, plan)
            # Tool diversity
            tool_diversity = len(set(step[0] for step in plan))  # Unique tools used
            # Entropy-based diversity
            entropy_score = self.logger.calculate_plan_entropy(plan)
            past_success = self.logger.get_model_success_rate(result["model_name"])

            return (
                0.4 * semantic_similarity +
                0.3 * tool_diversity +
                0.2 * entropy_score +
                0.1 * past_success
            )

        # Score all results and select the best
        scored_results = [(score_result(result), result) for result in results]
        best_result = max(scored_results, key=lambda x: x[0])[1]
        return best_result


# Implementation of Fallback Strategy
class ChainOfThoughtFallbackStrategy:
    def __init__(self, logger: Logging, models: Dict[str, BaseLLM]):
        self.logger = logger
        self.models = models

    def refine_plan(self, goal: str, context: Optional[str], task_type: Optional[str]) -> List[Tuple[str, str]]:
        """
        Generate a fallback plan using a meta-model with chain-of-thought generation.
        """
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


# Implementation of Plan Voting Strategy
class FragmentVotingStrategy:
    def __init__(self, logger: Logging):
        self.logger = logger

    def merge_or_vote(self, results: List[Dict]) -> List[Tuple[str, str]]:
        """
        Merge or vote on plans from multiple successful models.
        Rewards frequently reused fragments.
        """
        successful_plans = [result["plan"] for result in results if result["success"]]

        # Voting logic: Select the most common steps across plans
        fragment_scores = {}
        for plan in successful_plans:
            for step in plan:
                fragment_scores[step] = fragment_scores.get(step, 0) + 1

        # Reward fragments based on historical success
        for fragment, score in fragment_scores.items():
            historical_score = self.logger.get_fragment_success_rate(fragment)
            fragment_scores[fragment] += historical_score

        # Sort fragments by their scores
        merged_plan = sorted(fragment_scores.keys(), key=lambda step: fragment_scores[step], reverse=True)
        return merged_plan


# LLMRouter Implementation
class LLMRouter:
    def __init__(self, config: Dict[str, Dict], evaluation_strategy: EvaluationStrategy,
                 fallback_strategy: FallbackStrategy, voting_strategy: PlanVotingStrategy):
        """
        Initialize the router with a configuration of models and modular strategies.

        :param config: Dictionary with model names as keys and their parameters as values.
        :param evaluation_strategy: Strategy for evaluating results.
        :param fallback_strategy: Strategy for refining plans on failure.
        :param voting_strategy: Strategy for merging or voting on plans.
        """
        self.config = config
        self.models = {}
        self.logger = Logging()
        self.feedback_weights = {}  # Dynamic task-specific feedback weights
        self.task_profiles = {}  # Tracks task-specific performance profiles for models

        # Injected strategies
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
            # Add other model types here
        return self.models[name]

    def generate_plan(self, goal: str, context: Optional[str] = None, task_type: Optional[str] = None) -> List[Tuple[str, str]]:
        top_models = self._select_top_models(task_type, num_models=3)
        results = self._execute_in_parallel(top_models, goal, context)

        # If all models fail, attempt to refine the plan
        if not any(result["success"] for result in results):
            return self.fallback_strategy.refine_plan(goal, context, task_type)

        # If multiple models succeed, merge or vote on their plans
        if sum(1 for result in results if result["success"]) > 1:
            return self.voting_strategy.merge_or_vote(results)

        # Otherwise, evaluate and return the best single result
        best_result = self.evaluation_strategy.evaluate(results, goal, task_type)
        self._update_model_profiles(best_result["model_name"], task_type, success=True)
        return best_result["plan"]

    def _select_top_models(self, task_type: Optional[str], num_models: int = 3) -> List[str]:
        # Implementation for selecting top models (omitted for brevity)
        pass

    def _execute_in_parallel(self, models: List[str], goal: str, context: Optional[str]) -> List[Dict]:
        # Implementation for executing models in parallel (omitted for brevity)
        pass

    def _update_model_profiles(self, model_name: str, task_type: Optional[str], success: bool):
        # Implementation for updating model profiles (omitted for brevity)
        pass