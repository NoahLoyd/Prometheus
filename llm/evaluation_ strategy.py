from typing import List, Dict, Optional
from core.logging import Logging


class EvaluationStrategy:
    def evaluate(self, results: List[Dict], goal: str, task_type: Optional[str]) -> Dict:
        ...


class DefaultEvaluationStrategy(EvaluationStrategy):
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