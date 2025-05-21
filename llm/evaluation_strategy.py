from typing import Any, Dict, List, Optional

class EvaluationStrategy:
    """
    Abstract base class for evaluating LLM outputs.
    """
    def evaluate(self, outputs: List[Any], context: Optional[str] = None) -> Any:
        """
        Evaluate a list of outputs and select the best one.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("EvaluationStrategy.evaluate() must be implemented by subclasses.")

class DefaultEvaluationStrategy(EvaluationStrategy):
    """
    DefaultEvaluationStrategy selects and returns the result dictionary
    with the highest 'score' from a list of results. Designed for
    production use in AGI planning systems.

    Raises:
        ValueError: If the input list is empty.
        TypeError: If items are not dicts or 'score' is not numeric.
        KeyError: If item is missing a 'score' key.
    """

    def __init__(self, logger: Any = None):
        """
        Args:
            logger: Optional logger for diagnostics and warnings.
        """
        self.logger = logger

    def evaluate(self, results: List[Dict], context: Optional[str] = None) -> Dict:
        """
        Selects and returns the dictionary with the highest 'score' key.

        Args:
            results: List of result dictionaries, each with a numeric 'score' key.
            context: Optional context (unused in default strategy).

        Returns:
            The dictionary from results with the highest 'score'.

        Raises:
            ValueError: If results is empty.
            TypeError: If any result is not a dict or has a non-numeric score.
            KeyError: If any result is missing a 'score' key.
        """
        if not results:
            if self.logger:
                self.logger.warning("No results to evaluate.")
            raise ValueError("Cannot evaluate an empty list of results.")

        if self.logger:
            self.logger.info(f"Evaluating {len(results)} results using DefaultEvaluationStrategy.")

        best_result = None
        best_score = float('-inf')

        for idx, result in enumerate(results):
            if not isinstance(result, dict):
                if self.logger:
                    self.logger.error(f"Result at index {idx} is not a dict: {result}")
                raise TypeError(f"Result at index {idx} is not a dictionary.")
            if "score" not in result:
                if self.logger:
                    self.logger.error(f"Result at index {idx} missing 'score' key: {result}")
                raise KeyError(f"Result at index {idx} missing 'score' key.")
            score = result["score"]
            if not isinstance(score, (int, float)):
                if self.logger:
                    self.logger.error(f"Result at index {idx} has non-numeric score: {score}")
                raise TypeError(f"Result at index {idx} has non-numeric score.")

            if score > best_score:
                best_score = score
                best_result = result

        if best_result is None:
            # Should not happen, but fallback for extreme edge case
            if self.logger:
                self.logger.error("No valid results found after evaluation.")
            raise ValueError("No valid results found after evaluation.")

        return best_result
