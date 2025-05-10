from typing import List, Dict, Optional


class EvaluationStrategy:
    """
    Compares multiple model outputs and chooses the best based on task success, coherence, and relevance.
    """

    def evaluate(self, results: List[Dict], goal: str, task_type: Optional[str]) -> Dict:
        """
        Evaluate model outputs and select the best one.

        :param results: List of model output results.
        :param goal: The original goal or task.
        :param task_type: The type of task (e.g., 'reasoning', 'coding').
        :return: The best result dictionary.
        """
        successful_results = [result for result in results if result["success"]]
        if not successful_results:
            raise ValueError("No successful results to evaluate.")

        # Sort by relevance and coherence
        sorted_results = sorted(successful_results, key=lambda r: r.get("relevance", 0) + r.get("coherence", 0), reverse=True)
        return sorted_results[0]