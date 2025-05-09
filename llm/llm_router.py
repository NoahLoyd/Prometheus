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
            semantic_similarity = self.logger.calculate_semantic_similarity(goal, plan)
            tool_diversity = len(set(step[0] for step in plan))  # Unique tools used
            entropy_score = self.logger.calculate_plan_entropy(plan)
            past_success = self.logger.get_model_success_rate(result["model_name"])

            return (
                0.4 * semantic_similarity +
                0.3 * tool_diversity +
                0.2 * entropy_score +
                0.1 * past_success
            )

        scored_results = [(score_result(result), result) for result in results]
        best_result = max(scored_results, key=lambda x: x[0])[1]
        return best_result


# Implementation of Fallback Strategy
class ChainOfThoughtFallbackStrategy:
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


# Implementation of Plan Voting Strategy
class FragmentVotingStrategy:
    def __init__(self, logger: Logging):
        self.logger = logger

    def merge_or_vote(self, results: List[Dict]) -> List[Tuple[str, str]]:
        successful_plans = [result["plan"] for result in results if result["success"]]

        fragment_scores = {}
        for plan in successful_plans:
            for step in plan:
                fragment_scores[step] = fragment_scores.get(step, 0) + 1

        for fragment, score in fragment_scores.items():
            historical_score = self.logger.get_fragment_success_rate(fragment)
            fragment_scores[fragment] += historical_score

        merged_plan = sorted(fragment_scores.keys(), key=lambda step: fragment_scores[step], reverse=True)
        return merged_plan


# LLMRouter Implementation
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

    def _select_top_models(self, task_type: Optional[str], num_models: int = 3) -> List[str]:
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

        sorted_models = sorted(scores, key=scores.get, reverse=True)
        return sorted_models[:num_models]

    def _execute_in_parallel(self, models: List[str], goal: str, context: Optional[str]) -> List[Dict]:
        def execute_model(model_name: str) -> Dict:
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
            return [future.result() for future in futures]

    def _update_model_profiles(self, model_name: str, task_type: Optional[str], success: bool):
        key = (model_name, task_type)
        if key not in self.task_profiles:
            self.task_profiles[key] = {"success_count": 0, "failure_count": 0}

        if success:
            self.task_profiles[key]["success_count"] += 1
        else:
            self.task_profiles[key]["failure_count"] += 1

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

    def self_improve(self):
        """
        Periodically evaluates model and fragment performance to suggest updates.
        """
        underperforming_models = []
        for (model_name, task_type), profile in self.task_profiles.items():
            failure_ratio = profile["failure_count"] / (profile["success_count"] + profile["failure_count"] + 1e-6)
            if failure_ratio > 0.5:
                underperforming_models.append(model_name)

        self.logger.log_event("self_improve", f"Consider replacing models: {underperforming_models}")